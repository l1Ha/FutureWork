#!/usr/bin/env python3
"""
FutureWork Windows 便携发行包打包脚本 (Windows Portable Distribution Packager)
"""

from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DIST = ROOT / "dist"
WINDOWS_DIR = ROOT / "packaging" / "windows"


def create_windows_bundle() -> Path:
    DIST.mkdir(parents=True, exist_ok=True)
    bundle_name = "FutureWork-0.1.0-windows-x64"
    staging_dir = DIST / bundle_name

    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    # 1. 复制运行批处理与图标
    shutil.copy2(WINDOWS_DIR / "futurework.bat", staging_dir / "futurework.bat")
    shutil.copy2(WINDOWS_DIR / "futurework-hud.bat", staging_dir / "futurework-hud.bat")
    shutil.copy2(ROOT / "packaging" / "icon.ico", staging_dir / "futurework.ico")
    shutil.copy2(ROOT / "README.md", staging_dir / "README.md")
    shutil.copy2(ROOT / "LICENSE", staging_dir / "LICENSE")

    # 2. 复制 futurework 核心源码及 Web HUD 资源
    shutil.copytree(ROOT / "futurework", staging_dir / "futurework", dirs_exist_ok=True)

    # 3. 复制 examples
    shutil.copytree(ROOT / "examples", staging_dir / "examples", dirs_exist_ok=True)

    # 4. 生成 Windows 快速使用指引
    with open(staging_dir / "说明_WINDOWS.txt", "w", encoding="utf-8") as f:
        f.write("""======================================================================
  FutureWork Windows 便携发行版 (Portable Edition v0.1.0)
======================================================================

【极速启动方式】
1. 双击运行「futurework-hud.bat」：
   - 自动在后台启动 FutureWork 生产力核心，并在默认浏览器中弹出交互式 Web HUD 控制台。
   
2. 在命令行终端使用：
   - 打开 CMD 或 PowerShell，进入本解压目录
   - 运行: futurework.bat interactive  (进入沉浸式交互会话)
   - 运行: futurework.bat doctor       (检查系统适配器健康状态)
   - 运行: futurework.bat say "列出目录 ."

【环境要求】
- Windows 10 / Windows 11 (64位)
- Python 3.9+ 环境已安装并勾选添加到系统 PATH
- 依赖轻量核心库: pip install pydantic
======================================================================
""")

    # 5. 打包为 ZIP
    zip_path = DIST / f"{bundle_name}.zip"
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(staging_dir):
            for file in files:
                file_path = Path(root) / file
                arcname = file_path.relative_to(DIST)
                zf.write(file_path, arcname)

    print(f"✓ 成功生成 Windows 便携发行包：{zip_path} ({zip_path.stat().st_size // 1024} KB)")
    return zip_path


if __name__ == "__main__":
    create_windows_bundle()
