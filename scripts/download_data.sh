#!/usr/bin/env bash
set -euo pipefail
mkdir -p data
curl -L -o data/minidev.zip https://bird-bench.oss-cn-beijing.aliyuncs.com/minidev.zip
unzip -q -o data/minidev.zip -d data
rm data/minidev.zip
find data -name "mini_dev_sqlite.json"
