#!/usr/bin/env bash
# One command: raw extract -> clean data -> anomalies -> drivers -> backtest + forecast -> tests
set -e
cd "$(dirname "$0")"
[ -f dataset/student_data/train.csv ] || { echo "Put the raw CSVs in dataset/student_data/ first"; exit 1; }
echo "[1/5] Cleaning + unconstrained demand";   python src/clean.py
echo "[2/5] Drivers + intermittency";           python src/features.py
echo "[3/5] Anomaly register";                  python src/anomalies.py
echo "[4/5] Backtest + final forecast";         python src/models.py
echo "[5/5] Diagrams + tests";                    python src/generate_diagrams.py && python -m pytest tests -q
echo "Done. Dashboard: streamlit run dashboard/app.py"
