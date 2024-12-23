#!/bin/bash
set -e
PATH_TO_SAVE_TESTS=./fuzz_gcov_tests_2
BACKEND=torchdynamo
MODEL=torch
DEVICE=cpu
PARALLEL_NUM=16
LLVM_CONFIG=$(which llvm-config-14)
BS=10
TORCH_ROOT=/home/zhangzihan/pytorch-gcov
INST_LIBS="${TORCH_ROOT}/build/lib/libtorch.so ${TORCH_ROOT}/build/lib/libtorch_cpu.so"

python experiments/collect_python_coverage.py --root ${PATH_TO_SAVE_TESTS}        \
                                               --model_type ${MODEL}      \
                                              --backend_type ${BACKEND}  \
                                              --backend_target ${DEVICE} \
                                              --parallel ${PARALLEL_NUM}
#
#
#
# python experiments/process_profraws.py --root ${PATH_TO_SAVE_TESTS}       \
#                                       --llvm-config-path  ${LLVM_CONFIG} \
#                                       --batch-size        ${BS}          \
#                                       --instrumented-libs ${INST_LIBS}

# python experiments/viz_merged_cov.py -o results --ort --folders \
# ${PATH_TO_SAVE_TESTS}/coverage --tags "NNSmith" --pdf
