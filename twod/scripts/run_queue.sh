#!/bin/sh
# usage: run_queue.sh jobs.txt  -- starts the next job whenever fewer than 4 training processes run; skips finished jobs.
# each line: NAME|args[|DEPENDENCY_NAME]   (results in results/stage2b/NAME/; a dependent job waits for DEP's result.json)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
LOGDIR=${LOGDIR:-/tmp/q}; mkdir -p $LOGDIR
while IFS='|' read -r name args dep; do
  [ -z "$name" ] && continue
  [ -f results/stage2b/$name/result.json ] && continue
  if [ -n "$dep" ]; then while [ ! -f results/stage2b/$dep/result.json ]; do sleep 30; done; fi
  while [ $(pgrep -fc "python -u stage2b_train") -ge 4 ]; do sleep 20; done
  setsid nohup python -u stage2b_train.py --name $name $args >> $LOGDIR/$name.log 2>&1 < /dev/null &
  sleep 5
done < $1
