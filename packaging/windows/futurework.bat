@echo off
setlocal
set "DIR=%~dp0"
set "PYTHONPATH=%DIR%;%PYTHONPATH%"

:: 检查 Python 环境
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [错误] 未检测到 Python 环境，请安装 Python 3.9+ 并将其添加至 PATH。
    pause
    exit /b 1
)

python -m futurework %*
endlocal
