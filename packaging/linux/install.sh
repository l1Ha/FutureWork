#!/usr/bin/env bash
# FutureWork Linux 快捷安装脚本（用户级安装，无需 sudo）
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$DIR/futurework"

if [ ! -f "$BIN" ]; then
    echo "错误：未找到可执行文件 $BIN"
    exit 1
fi

chmod +x "$BIN"

# 1. 复制二进制到 ~/.local/bin
mkdir -p "$HOME/.local/bin"
cp "$BIN" "$HOME/.local/bin/futurework"
echo "✓ 已将可执行文件安装至 $HOME/.local/bin/futurework"

# 2. 安装图标
mkdir -p "$HOME/.local/share/icons/hicolor/512x512/apps"
if [ -f "$DIR/futurework.png" ]; then
    cp "$DIR/futurework.png" "$HOME/.local/share/icons/hicolor/512x512/apps/futurework.png"
    echo "✓ 已安装应用图标至 $HOME/.local/share/icons"
fi

# 3. 安装桌面快捷方式
mkdir -p "$HOME/.local/share/applications"
if [ -f "$DIR/futurework.desktop" ]; then
    cp "$DIR/futurework.desktop" "$HOME/.local/share/applications/futurework.desktop"
    echo "✓ 已安装桌面应用快捷方式至 $HOME/.local/share/applications/futurework.desktop"
fi

echo ""
echo "安装完成！你现在可以通过以下方式使用 FutureWork："
echo "  1. 在应用启动器中搜索并点击 'FutureWork'"
echo "  2. 在终端直接输入 'futurework hud' 或 'futurework interactive'"
