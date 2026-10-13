@echo off
rem LiquidAion Neural Observatory: download-and-run launcher for Windows (double-click).
rem First run: installs uv (pinned) and Python 3.12 into .tools\, then PyTorch and dependencies into
rem .liquidaion-env\ (CUDA build if an NVIDIA GPU is visible, else CPU). Options: see scripts\bootstrap.py.
setlocal
set "ROOT=%~dp0"
set "UV_VERSION=0.12.17"
set "UV=%ROOT%.tools\uv.exe"
set "UV_PYTHON_INSTALL_DIR=%ROOT%.tools\python"
set "UV_CACHE_DIR=%ROOT%.tools\cache"
if not exist "%UV%" (
  echo [LiquidAion] Installing uv %UV_VERSION% into %ROOT%.tools ^(first run only^)
  powershell -NoProfile -ExecutionPolicy Bypass -Command "$env:UV_INSTALL_DIR='%ROOT%.tools'; $env:UV_NO_MODIFY_PATH='1'; irm https://astral.sh/uv/%UV_VERSION%/install.ps1 | iex"
  if not exist "%UV%" goto :uvfailed
)
set "LIQUIDAION_UV=%UV%"
"%UV%" run --no-project --python 3.12 python "%ROOT%scripts\bootstrap.py" %*
set "CODE=%ERRORLEVEL%"
if not "%CODE%"=="0" (
  echo.
  echo [LiquidAion] Finished with exit code %CODE%. The messages above explain the problem.
  if not defined LIQUIDAION_NO_PAUSE pause
)
exit /b %CODE%
:uvfailed
echo [LiquidAion] uv could not be installed. Check the internet connection and try again.
if not defined LIQUIDAION_NO_PAUSE pause
exit /b 1
