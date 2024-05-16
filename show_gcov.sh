#!/bin/bash
set -e
FUZZ_ROOT=./fuzz_gcov_report_2
FUZZ_TIME="10min"
PATH_TO_SAVE_TESTS=./fuzz_gcov_tests_2
BACKEND=torchdynamo
MODEL=torch
DEVICE=cpu
PARALLEL_NUM=16
LLVM_CONFIG=$(which llvm-config-14)
BS=10
TORCH_ROOT=/home/zhangzihan/pytorch-gcov
INST_LIBS="${TORCH_ROOT}/build/lib/libtorch.so ${TORCH_ROOT}/build/lib/libtorch_cpu.so"

if [[ "$CONDA_DEFAULT_ENV" != *gcov* ]]; then
    echo "The CONDA_DEFAULT_ENV does not contain 'gcov'."
    exit 1
fi
if [ ! -d "$FUZZ_ROOT" ]; then
    echo "Error: Directory $FUZZ_ROOT already exists. please modify it in the bash script."
    exit 1
fi
# python experiments/evaluate_models.py --root ${PATH_TO_SAVE_TESTS}        \
#                                                --model_type ${MODEL}      \
#                                               --backend_type ${BACKEND}  \
#                                               --backend_target ${DEVICE} \
#                                               --parallel ${PARALLEL_NUM}
#
#
#
# python experiments/process_profraws.py --root ${PATH_TO_SAVE_TESTS}       \
#                                       --llvm-config-path  ${LLVM_CONFIG} \
#                                       --batch-size        ${BS}          \
#                                       --instrumented-libs ${INST_LIBS}

python experiments/viz_merged_cov.py -o results --ort --folders \
${PATH_TO_SAVE_TESTS}/coverage --tags "NNSmith" --pdf
