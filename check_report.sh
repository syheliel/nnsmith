#!/bin/bash
report_root=$1
if [[ -z "$1" ]]; then
    echo "usage: check_report.sh <fuzz_report_folder>"
    exit 1  # 如果输入为空，则退出脚本，并将退出状态码设置为 1
else
    echo "输入非空：$1"
fi
RESULT_PATH="${report_root}_result.txt"
for dir in $(realpath $1/*); do
  echo "start to deal with $dir"
  TARGET_PATH=$(realpath "$dir/syn.py")
  python3 $TARGET_PATH 2>&1
  if [ $? -ne 0 ]; then
    echo "An error occurred, the script path is $TARGET_PATH"
    echo $TARGET_PATH >> $RESULT_PATH
else
    echo "can't reproduce the error"
fi
done
  echo "generate result file at $RESULT_PATH"
