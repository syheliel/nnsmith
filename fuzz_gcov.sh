#!/bin/bash
set -e
FUZZ_ROOT=./fuzz_gcov_report_2
FUZZ_TIME="10min"
PATH_TO_SAVE_TESTS=./fuzz_gcov_tests_2

if [[ "$CONDA_DEFAULT_ENV" != *gcov* ]]; then
    echo "The CONDA_DEFAULT_ENV does not contain 'gcov'."
    exit 1
fi
if [ -d "$FUZZ_ROOT" ]; then
    echo "Error: Directory $FUZZ_ROOT already exists. please modify it in the bash script."
    exit 1
fi
nnsmith.fuzz fuzz.time=$FUZZ_TIME model.type=torch backend.type=torchdynamo fuzz.root=$FUZZ_ROOT debug.viz=true mgen.max_nodes=20 fuzz.save_test=${PATH_TO_SAVE_TESTS}
