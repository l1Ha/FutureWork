@echo off
setlocal
set "DIR=%~dp0"

:: 优先使用捆绑的独立 exe，降级到 python -m
if exist "%DIR%futurework.exe" (
    "%DIR%futurework.exe" %*
    goto :end
)

where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [错误] 未找到 futurework.exe 且未安装 Python 3.9+。
    exit /b 1
)
set "PYTHONPATH=%DIR%;%PYTHONPATH%"
python -m futurework %*

:end
endlocal
