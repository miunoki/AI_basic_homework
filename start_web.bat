@echo off
setlocal

cd /d "%~dp0"
set PYTHONIOENCODING=utf-8

echo Starting the campus Q&A web app...
echo The browser will open automatically after the server is ready.
echo Keep this window open while using the app.
echo.

where python >nul 2>nul
if %errorlevel%==0 (
    python app.py
) else (
    py -3 app.py
)

echo.
echo The app has stopped. If startup failed, check that Python and dependencies are installed.
pause
