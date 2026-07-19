@echo off
set "PYTHONPATH=%cd%"
echo Setting PYTHONPATH to "%PYTHONPATH%"

echo Starting Federated Learning Server...
start "FL Server" cmd /k "python fl/server.py"

timeout /t 5 /nobreak

echo Starting FL Client 0...
start "FL Client 0" cmd /k "python fl/client_app.py 0"

echo Starting FL Client 1...
start "FL Client 1" cmd /k "python fl/client_app.py 1"

echo Starting FL Client 2...
start "FL Client 2" cmd /k "python fl/client_app.py 2"

echo All processes launched! You should see 4 new windows.
pause
