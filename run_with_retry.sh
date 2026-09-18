#!/bin/bash
# Self-healing runner: relaunches run_full_matrix.py if it exits for any reason
# (crash, whatever killed the kernel before) before all 26 datasets are complete.
# The resume-skip guard inside the script means each relaunch picks up where the
# last one left off, never redoing a completed dataset.
cd "/Users/sanju/Desktop/AI new papoer"
source .venv_notebook/bin/activate

while true; do
  DONE=$(python3 -c "
import pandas as pd
try:
    df = pd.read_csv('results/raw_runs.csv')
    g = df.groupby('dataset')['seed'].nunique()
    print((g >= 10).sum())
except Exception:
    print(0)
")
  echo "$(date): $DONE / 26 datasets complete"
  if [ "$DONE" -ge 26 ]; then
    echo "$(date): ALL DATASETS COMPLETE"
    break
  fi
  echo "$(date): launching run_full_matrix.py"
  python3 run_full_matrix.py
  EXIT_CODE=$?
  echo "$(date): run_full_matrix.py exited with code $EXIT_CODE -- checking progress and relaunching if needed"
  sleep 5
done
