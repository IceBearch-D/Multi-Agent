#!/usr/bin/env bash
# 后台看门狗：轮询 10.0.131.251:9005，服务一旦可达即运行 AlgoSolver 全流程。
# 受总时长 5 小时上限约束（与任务要求一致）。
set -u
cd /home/icebearch/Multi-Agent

PY=/home/icebearch/Multi-Agent/.venv/bin/python
DEADLINE=$(( $(date +%s) + 5*3600 ))
RUNS=0

probe() {
  "$PY" - <<'PY'
import urllib.request, urllib.error
BASE="http://10.0.131.251:9005/v1"
for p in ("/models", "/health", "/"):
    try:
        r = urllib.request.urlopen(BASE + p, timeout=6)
        if r.status < 500:
            print("UP"); break
    except urllib.error.HTTPError as e:
        if e.code < 500:
            print("UP"); break
    except Exception:
        pass
else:
    print("DOWN")
PY
}

while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  UP=$(probe)
  if [ "$UP" = "UP" ]; then
    echo "[$(date)] 服务可达，开始运行解题流水线" | tee -a run_log.txt
    "$PY" src/main.py --problems problems --max-iter 4 --output report.json >> run_log.txt 2>&1
    CODE=$?
    echo "[$(date)] 运行结束 code=$CODE" | tee -a run_log.txt
    if [ "$CODE" -eq 0 ]; then
      echo "全部通过，停止看门狗。" | tee -a run_log.txt
      break
    elif [ "$CODE" -eq 1 ]; then
      RUNS=$((RUNS + 1))
      if [ "$RUNS" -ge 3 ]; then
        echo "已尝试 $RUNS 次仍未全通过，停止。" | tee -a run_log.txt
        break
      fi
      echo "第 $RUNS 次未全通过，120s 后重试…" | tee -a run_log.txt
      sleep 120
    else
      # code 2 = 服务判定不可用，继续轮询
      sleep 60
    fi
  else
    echo "[$(date)] 服务不可用(502)，60s 后重试" | tee -a run_log.txt
    sleep 60
  fi
done
echo "[$(date)] 看门狗结束" | tee -a run_log.txt
