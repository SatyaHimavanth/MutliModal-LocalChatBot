@echo off
setlocal

for %%I in ("%~dp0..") do set "APP_ROOT=%%~fI"
set "LLAMA_SERVER=%APP_ROOT%\vendor\llama.cpp\windows-x86_64\llama-server.exe"

if not exist "%LLAMA_SERVER%" (
  echo llama-server.exe was not found:
  echo   "%LLAMA_SERVER%"
  exit /b 1
)

if not exist "%APP_ROOT%\models" (
  echo The models directory was not found:
  echo   "%APP_ROOT%\models"
  exit /b 1
)

pushd "%APP_ROOT%" || exit /b 1
echo Starting the local llama.cpp model router on http://127.0.0.1:8080
echo Press Ctrl+C to stop the server.

"%LLAMA_SERVER%" --models-dir ".\models" --host 127.0.0.1 --port 8080 --models-max 2 --sleep-idle-seconds 300
set "SERVER_EXIT=%ERRORLEVEL%"

popd
exit /b %SERVER_EXIT%
