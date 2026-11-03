@echo off
setlocal
set "DIR=%~dp0"

echo ======================================================================
echo    FutureWork 面向未来人机自然交互生产力核心 (Web HUD)
echo ======================================================================

:: 延迟 2 秒后在默认浏览器打开控制台
start "" cmd /c "timeout /t 2 /nobreak >nul && start http://127.0.0.1:8765"

:: 优先使用捆绑的独立 exe（零 Python 依赖），降级到 python -m
if exist "%DIR%futurework.exe" (
    echo 检测到独立可执行文件，直接启动...
    "%DIR%futurework.exe" hud --host 127.0.0.1 --port 8765
    goto :end
)

where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [错误] 未找到 futurework.exe 且未安装 Python 3.9+。
    pause
    exit /b 1
)
set "PYTHONPATH=%DIR%;%PYTHONPATH%"
python -m futurework hud --host 127.0.0.1 --port 8765

:end
endlocal
