# --------------------
# /home/zhangzihan/nnsmith/fuzz_report_3/bug-Symptom.EXCEPTION-Stage.EXECUTION-1/model.pth

import numpy as np
import torch

# Model definition
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.v2_0 = torch.nn.Parameter(torch.empty([1, 22, 51], dtype=torch.int64), requires_grad=False)

    def forward(self, _args):
        v2_0 = self.v2_0
        getitem = _args
        max_1 = getitem.max(0)
        getattr_1 = max_1.values
        max_2 = torch.max(getitem, v2_0)
        return (getattr_1, max_2)

m = M()

inp =  torch.from_numpy(np.zeros((22, 51),dtype=np.int64))
m(inp) # this line is OK
opt = torch.compile(m, fullgraph=True, backend='inductor', mode=None)
opt(inp) # this line will crash

# Eager run
# m_out = m(*[torch.from_numpy(v).to('cpu') for v in inp])
# m_out = [v.cpu().detach() for v in m_out] # torch2numpy
# m_out = [v.resolve_conj().numpy() if v.is_conj() else v.numpy() for v in m_out] # torch2numpy

# # Compiled run
# opt_out = opt(*[torch.from_numpy(v).to('cpu') for v in inp])
# opt_out = [v.cpu().detach() for v in opt_out] # torch2numpy
# opt_out = [v.resolve_conj().numpy() if v.is_conj() else v.numpy() for v in opt_out] # torch2numpy

# # Differential testing
# for i, (l, r) in enumerate(zip(m_out, opt_out)):
#     np.testing.assert_allclose(l, r, rtol=1e-2, atol=1e-3, err_msg=f"Result mismatch @ index {i}")

# # --------------------

# AOTInductor error: Unsupported reduction type from torch.float32 to torch.int64