@echo off
setlocal
set "DIR=%~dp0"
set "PYTHONPATH=%DIR%;%PYTHONPATH%"

echo ======================================================================
echo    FutureWork 面向未来人机自然交互生产力核心 (Web HUD)
echo ======================================================================
echo 正在启动本地服务并打开浏览器控制台...

:: 检查 Python 环境
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [错误] 未检测到系统 Python，请安装 Python 3.9+。
    pause
    exit /b 1
)

:: 延迟 1.5 秒后在默认浏览器打开控制台
start "" cmd /c "timeout /t 2 /nobreak >nul && start http://127.0.0.1:8765"

:: 启动 Web HUD 服务器
python -m futurework hud --host 127.0.0.1 --port 8765

endlocal
