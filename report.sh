#!/bin/bash
set -e
report_root=$1
if [[ -z "$1" ]]; then
    echo "输入为空，脚本将退出。"
    exit 1  # 如果输入为空，则退出脚本，并将退出状态码设置为 1
else
    echo "输入非空：$1"
fi
for dir in $(realpath $1/*); do
  echo "start to deal with $dir"
  MODEL_PATH="$dir/model.pth"
  PYTHON_PATH="$dir/syn.py"
  nnsmith.report_syn backend.type="pt2 backend@inductor" model.type=torch model.path=$MODEL_PATH > $PYTHON_PATH
  echo "generate file at $PYTHON_PATH"
done
