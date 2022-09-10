"""
Class members of AbsOpBase like in_dtypes, out_dtypes, etc. are just the superset
of the valid domain that do not take the upcoming inference engine's operator
availability into account. For example, older versions of TVM may not support trilu
and TensorRT does not accept a 2D Pool whose kernel is larger than 300. Therefore,
to narrow those specifications, we look at the following methods:
- Identifier: Model and BackendFactory
1. Single-operator specification testing: Iterate over possible operator instances
   of available `data types` (and `ranks`), and try to kick out failing ones (loggings
   will be kept to see if it is just an unimplemented feature or a bug).
2. Constraint extension: for a BackendFactory, it can add more constraints to an operator.
   This is useful for cases like TensorRT where the kernel size is limited to < 300.
- HOWTO:
1. Single-operator potential data types (and ranks) are serializable as a JSON file.
2. Constraint extensions are defined as Python codes. That said, we can just overload the
   operator specifactions in backend (see nnsmith/abstract/extension).
"""


import os
from copy import deepcopy
from dataclasses import dataclass
from os import PathLike
from typing import Dict, List, Type

import z3
from appdirs import user_cache_dir
from omegaconf import OmegaConf

from nnsmith.abstract.dtype import DType
from nnsmith.abstract.op import AbsOpBase, AbsTensor, Constant, Input, concretize_op
from nnsmith.backends import BackendFactory
from nnsmith.materialize import Instruction, Model, Schedule, TestCase
from nnsmith.util import fail_print, note_print, succ_print

NNSMITH_CACHE_DIR = user_cache_dir("nnsmith")


def get_cache_name(model_cls: Type[Model], factory: BackendFactory) -> str:
    return f"{model_cls.__name__}_{factory.system_name}_{factory.device}"


@dataclass
class OpConfig:
    in_dtypes: List[List[DType]]
    out_dtypes: List[List[DType]]


def _make_single_op_schedules(
    op: AbsOpBase, ishapes, available_idtypes
):  # List<Tup<DTypeComb,DTypeComb,Sched>>
    """Given a concretized op, return a schedule for it."""
    schedule_list = []
    input_keys = list(range(len(ishapes)))
    for idtype_group in available_idtypes:
        key2type = {}
        instr = []
        inputs = []

        for idx, (ishape, idtype) in enumerate(zip(ishapes, idtype_group)):
            inp = Input(dim=len(ishape))
            inp.abs_tensor = AbsTensor(shape=ishape, dtype=idtype)
            instr.append(
                Instruction(
                    op=inp,
                    inputs=[],
                    outputs=[idx],
                )
            )
            key2type[input_keys[idx]] = inp.abs_tensor
            inputs.append(inp.abs_tensor)

        this_op = deepcopy(op)
        outputs = this_op.checked_type_transfer(inputs)
        this_op.input_like = inputs

        leaf_keys = list(range(len(inputs), len(inputs) + len(outputs)))

        instr.append(
            Instruction(
                op=this_op,
                inputs=input_keys,
                outputs=leaf_keys,
            )
        )

        for key, aten in zip(leaf_keys, outputs):
            key2type[key] = aten

        schedule_list.append(
            (
                idtype_group,
                tuple([out.dtype for out in outputs]),
                Schedule(instr, input_keys, leaf_keys, key2type),
            )
        )
    return schedule_list


def infer_topset_from_scratch(
    model_cls: Model, factory: BackendFactory, op_types=None, verbose=True
) -> Dict[str, OpConfig]:
    if op_types is None:
        op_types = model_cls.operators()

    topset = {}

    n_ops = len(op_types)
    for idx, node_t in enumerate(op_types):
        if node_t is Input or node_t is Constant:
            continue

        if verbose:
            note_print(f"[{idx + 1} / {n_ops}] ===> Trying {node_t}")

        available_idtypes = node_t.in_dtypes

        if not available_idtypes:
            continue

        op_param_n = node_t.get_num_var_param()
        op_params = [z3.Int("v%s-%s" % (idx, k)) for k in range(op_param_n)]
        op = node_t(*op_params)

        solver = z3.Solver()

        inputs = []
        for i, ranks in enumerate(op.inp_ranks):
            if op.same_inp_dims and inputs:
                rank = inputs[0].ndims
            else:
                rank = min(ranks)
            shape = AbsTensor(
                shape=[z3.Int("s%s" % (k)) for k in range(rank)],
                dtype=available_idtypes[0][i],
            )
            inputs.append(shape)
            solver.add(*shape.gt_zero())

        solver.add(*op.checked_requires(inputs))

        # solve
        assert solver.check() == z3.sat, f"Cannot solve the problem in {node_t}"
        m = solver.model()

        # rewrite
        concrete_op = concretize_op(op, m)
        concre_input_shapes = []
        for inp in inputs:
            shape = []
            for s in inp.shape:
                shape.append(m.eval(s).as_long())
            concre_input_shapes.append(shape)

        schedule_sets = _make_single_op_schedules(
            concrete_op, concre_input_shapes, available_idtypes
        )

        schedule_sets = [
            sset
            for sset in schedule_sets
            if set(factory.skip_dtypes()).isdisjoint(sset[0] + sset[1])
        ]

        op_itypes = set()
        op_otypes = set()

        for itypes, otypes, sched in schedule_sets:
            model = model_cls.from_schedule(sched)
            # Test compilation + simple inference;
            out = factory.make_testcase(model)
            if isinstance(out, TestCase):  # Pass
                if verbose:
                    succ_print(
                        f"=====> [Success] at {concrete_op}({itypes}) => {otypes}"
                    )
                op_itypes.add(itypes)
                op_otypes.add(otypes)
            else:  # Fail
                if verbose:
                    fail_print(
                        f"=====> [Failure] at {concrete_op}({itypes}) => {otypes}"
                    )
                    fail_print(f"{out.log}")

        if op_itypes:
            topset[op.name()] = OpConfig(
                in_dtypes=list(op_itypes), out_dtypes=list(op_otypes)
            )

    return topset


def load_topset(topset_path) -> Dict[str, OpConfig]:
    conf = OmegaConf.load(topset_path)["topset"]
    ret = {}
    # cvt str -> DType
    for k, v in conf.items():
        ret[k] = OpConfig(
            in_dtypes=[tuple([DType[t] for t in dtypes]) for dtypes in v["in_dtypes"]],
            out_dtypes=[
                tuple([DType[t] for t in dtypes]) for dtypes in v["out_dtypes"]
            ],
        )
    return ret


def dump_topset(topset: Dict[str, OpConfig], path: PathLike):
    OmegaConf.save({"topset": topset}, path)


def load_topset_from_auto_cache(
    model_cls: Model, factory: BackendFactory, verbose=True
) -> Dict[str, OpConfig]:
    cache_path = os.path.join(
        NNSMITH_CACHE_DIR, get_cache_name(model_cls, factory) + ".yaml"
    )
    # mkdir -p NNSMITH_CACHE_DIR
    if not os.path.exists(NNSMITH_CACHE_DIR):
        os.makedirs(NNSMITH_CACHE_DIR)
    if os.path.exists(cache_path):
        if verbose:
            succ_print(f"Loading topset from {cache_path}.")
            note_print("To regenerate the topset, delete the cache file and restart.")
            note_print(f"rm {cache_path}")
        return load_topset(cache_path)
    else:
        if verbose:
            note_print(f"Inferring topset from scratch and cache it to {cache_path}.")
        opset = infer_topset_from_scratch(model_cls, factory)
        dump_topset(opset, cache_path)
        return opset


def opset_from_auto_cache(model_cls: Model, factory: BackendFactory, verbose=True):
    topset_config = load_topset_from_auto_cache(model_cls, factory, verbose)
    opset = model_cls.operators()
    for i, op in enumerate(opset):
        op.in_dtypes = topset_config[op.name()].in_dtypes
        op.out_dtypes = topset_config[op.name()].out_dtypes
        opset[i] = op
    # TODO(@ganler): make patch.
    return opset