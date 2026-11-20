@echo off
echo ===============================================================================
echo   CELERIS DEMAND PLANNING CONTROL TOWER - AUTOMATED REPRODUCTION PIPELINE
echo ===============================================================================

echo [1/5] Running Data Cleaning and Unconstrained Demand Reconstruction...
python src/clean.py
if %ERRORLEVEL% NEQ 0 ( echo ERROR in src/clean.py & exit /b %ERRORLEVEL% )

echo [2/5] Running Anomaly Detection and Market Signals Engine...
python src/anomalies.py
if %ERRORLEVEL% NEQ 0 ( echo ERROR in src/anomalies.py & exit /b %ERRORLEVEL% )

echo [3/5] Synthesizing 75 Domain, Intermittent, and Macro Features...
python src/features.py
if %ERRORLEVEL% NEQ 0 ( echo ERROR in src/features.py & exit /b %ERRORLEVEL% )

echo [4/5] Training Dual-Tree Ensemble, Quantiles, and Submission Generation...
python src/models.py
if %ERRORLEVEL% NEQ 0 ( echo ERROR in src/models.py & exit /b %ERRORLEVEL% )

echo [5/5] Running Solution Verification & Integrity Tests...
python tests/test_solution.py
if %ERRORLEVEL% NEQ 0 ( echo ERROR in tests/test_solution.py & exit /b %ERRORLEVEL% )

echo ===============================================================================
echo   SUCCESS! All 5 pipeline stages and verification tests passed.
echo   Launch dashboard: streamlit run dashboard/app.py
echo ===============================================================================
