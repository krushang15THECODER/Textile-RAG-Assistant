@echo off
echo =====================================================================
echo  LabhTex Textile Intelligence RAG Platform - Local Startup
echo =====================================================================
echo.
echo Starting FastAPI Backend on http://127.0.0.1:8000 ...
start "LabhTex Backend (FastAPI)" cmd /k ".\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload"

timeout /t 3 /nobreak >nul

echo Starting Streamlit Frontend on http://localhost:8501 ...
start "LabhTex Frontend (Streamlit)" cmd /k ".\.venv\Scripts\streamlit.exe run app.py --server.port 8501"

echo.
echo Both services have been launched!
echo Access the frontend at: http://localhost:8501
echo API documentation at:   http://127.0.0.1:8000/docs
echo =====================================================================
pause
