#!/usr/bin/env bash
# AlgoSolver 一键运行脚本
set -u
cd "$(dirname "$0")"

# 激活虚拟环境（若存在）
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

# 加载 .env（若存在）
if [ -f .env ]; then
    set -a
    . ./.env
    set +a
fi

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

python src/main.py --problems problems --max-iter 4 --output report.json "$@"
