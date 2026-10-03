#!/usr/bin/env bash
# FutureWork Linux 快捷启动脚本
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$DIR/futurework"

if [ ! -f "$BIN" ]; then
    echo "错误：未找到可执行文件 $BIN"
    exit 1
fi

chmod +x "$BIN"
echo "正在启动 FutureWork 生产力核心 Web HUD..."
"$BIN" hud "$@"
