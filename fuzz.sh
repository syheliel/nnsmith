#!/bin/bash
set -e
FUZZ_ROOT=./fuzz_report_6
FUZZ_TIME="300min"
extra_flags_array=($extra_flags)
if [ -d "$FUZZ_ROOT" ]; then
    echo "Error: Directory $FUZZ_ROOT already exists. please modify it in the bash script."
    exit 1
fi
nnsmith.fuzz fuzz.time=$FUZZ_TIME model.type=torch backend.type=torchdynamo fuzz.root=$FUZZ_ROOT debug.viz=true mgen.max_nodes=20
