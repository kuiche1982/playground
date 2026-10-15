#!/bin/bash
# supervisor_bigtour.sh: 主进程(实验8)被环境回收后自动重启；全部 run 跑完（最终图生成）才退出
# 断点续跑由 multi_bug_genome.py --bigtour 自身保证（bigtour_runs/*.json 跳过已完成）
cd /home/user/Doubao/chats/25419664949684994/playground || exit 1
while true; do
  if [ -f multi_bug_bigtour_result.png ]; then
    echo "[$(date '+%F %T')] 实验8完成：最终图已生成，监督退出" >> bigtour_log.txt
    break
  fi
  if ! pgrep -f "multi_bug_genome.py --bigtour" > /dev/null; then
    echo "[$(date '+%F %T')] 主进程不在，重启实验..." >> bigtour_log.txt
    python3 -u multi_bug_genome.py --bigtour >> bigtour_log.txt 2>&1
    echo "[$(date '+%F %T')] 主进程退出(exit=$?)，30秒后检查..." >> bigtour_log.txt
  fi
  sleep 30
done
