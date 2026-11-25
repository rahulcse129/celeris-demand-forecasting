@echo off
REM One command: raw extract -> clean data -> anomalies -> drivers -> backtest + forecast -> tests
cd /d %~dp0
if not exist dataset\student_data\train.csv ( echo Put the raw CSVs in dataset\student_data\ first & exit /b 1 )
echo [1/5] Cleaning + unconstrained demand
python src/clean.py || exit /b 1
echo [2/5] Drivers + intermittency
python src/features.py || exit /b 1
echo [3/5] Anomaly register
python src/anomalies.py || exit /b 1
echo [4/5] Backtest + final forecast
python src/models.py || exit /b 1
echo [5/5] Tests
python -m pytest tests -q || exit /b 1
echo Done. Dashboard: streamlit run dashboard/app.py
