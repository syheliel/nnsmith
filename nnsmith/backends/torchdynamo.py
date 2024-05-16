import os
import warnings
from typing import Dict, List, Tuple

import numpy as np
import torch
from multipledispatch import dispatch

from nnsmith.backends.factory import BackendCallable, BackendFactory
from nnsmith.materialize.torch import TorchModel, numpify

# Check https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
# for more PyTorch-internal options.


class TorchDynamo(BackendFactory):
    def __init__(self, target="cpu", optmax: bool = False, **kwargs):
        super().__init__(target, optmax)
        if self.target == "cpu":
            self.device = torch.device("cpu")
        elif self.target == "cuda":
            self.device = torch.device("cuda")
            torch.backends.cudnn.benchmark = True
        else:
            raise ValueError(f"Unknown {target=}. Only `cpu` and `cuda` are supported.")

    @property
    def system_name(self) -> str:
        return "torchdynamo"

    @dispatch(TorchModel)
    def make_backend(self, model: TorchModel) -> BackendCallable:
        torch_net = model.torch_model.to(self.device)
        trace_inp = [ts.to(self.device) for ts in torch_net.get_random_inps().values()]

        do_grad_check = model.needs_grad_check()

        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore", category=torch.jit.TracerWarning)
        if do_grad_check:
            torch_net = torch_net.train()
        else:
            torch_net = torch_net.eval()

        compiled = torch.compile(torch_net)

        def closure(inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
            nonlocal do_grad_check
            input_ts = [torch.from_numpy(v).to(self.device) for _, v in inputs.items()]
            if do_grad_check:
                outputs: List[torch.Tensor] = compiled(*input_ts)
                params = {k: v for k, v in compiled.named_parameters()}
                ret = {}

                for name, output in zip(torch_net.output_like.keys(), outputs):
                    ret[name] = numpify(output)
                    if output.requires_grad:
                        # get Vector-Jacobian product
                        out_grad = torch.autograd.grad(
                            outputs=output,
                            inputs=params.values(),
                            grad_outputs=torch.ones_like(output),
                            retain_graph=True,
                            allow_unused=True,
                        )
                        for k, v in zip(params.keys(), out_grad):
                            ret[name + "_vjp_" + k] = numpify(v)
            else:
                with torch.no_grad():
                    outputs: Tuple[torch.Tensor] = compiled(*input_ts)
                ret = {k: numpify(v) for k, v in zip(torch_net.output_like, outputs)}
            return ret

        return closure

    @property
    def import_libs(self) -> List[str]:
        return ["import torch"]

    def emit_compile(self, opt_name: str, mod_name: str, inp_name: str) -> str:
        return f"{opt_name} = torch.jit.trace({mod_name}, [torch.from_numpy(v).to('{self.device.type}') for v in {inp_name}])"

    def emit_run(self, out_name: str, opt_name: str, inp_name: str) -> str:
        return f"""{out_name} = {opt_name}(*[torch.from_numpy(v).to('{self.device.type}') for v in {inp_name}])
{out_name} = [v.cpu().detach() for v in {out_name}] # torch2numpy
{out_name} = [v.resolve_conj().numpy() if v.is_conj() else v.numpy() for v in {out_name}] # torch2numpy"""
