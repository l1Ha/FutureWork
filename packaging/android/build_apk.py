#!/usr/bin/env python3
"""
FutureWork Android APK 自动化独立构建器 (Standalone Android APK Builder)

设计目标：
- 零笨重 Android Studio 与 native JDK 依赖，仅依赖任何标准 JRE (java) 与 Python 3
- 自动按需下载并缓存官方编译组件 (aapt2, ecj, r8/d8, android.jar, uber-apk-signer)
- 产出经过 V1+V2+V3 签名与 zipalign 对齐、可在真机与模拟器直接安装运行的 Release APK
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

# 编译组件官方下载镜像
AAPT2_JAR_URL = "https://dl.google.com/dl/android/maven2/com/android/tools/build/aapt2/8.2.2-10154469/aapt2-8.2.2-10154469-linux.jar"
ECJ_JAR_URL = "https://repo1.maven.org/maven2/org/eclipse/jdt/ecj/3.26.0/ecj-3.26.0.jar"
R8_JAR_URL = "https://dl.google.com/dl/android/maven2/com/android/tools/r8/8.2.33/r8-8.2.33.jar"
ANDROID_JAR_URL = "https://raw.githubusercontent.com/Sable/android-platforms/master/android-30/android.jar"
UBER_SIGNER_URL = "https://github.com/patrickfav/uber-apk-signer/releases/download/v1.3.0/uber-apk-signer-1.3.0.jar"

CACHE_DIR = Path.home() / ".cache" / "futurework_android_tools"


def download_file(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"正在下载编译组件：{dest.name} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "FutureWork-Builder/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as out:
        shutil.copyfileobj(resp, out)
    print(f"✓ 完成下载：{dest.name} ({dest.stat().st_size // 1024} KB)")


def prepare_tools() -> tuple[Path, Path, Path, Path, Path]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    aapt2_jar = CACHE_DIR / "aapt2.jar"
    aapt2_bin = CACHE_DIR / "aapt2"
    ecj_jar = CACHE_DIR / "ecj.jar"
    r8_jar = CACHE_DIR / "r8.jar"
    android_jar = CACHE_DIR / "android.jar"
    signer_jar = CACHE_DIR / "uber-apk-signer.jar"

    download_file(AAPT2_JAR_URL, aapt2_jar)
    download_file(ECJ_JAR_URL, ecj_jar)
    download_file(R8_JAR_URL, r8_jar)
    download_file(ANDROID_JAR_URL, android_jar)
    download_file(UBER_SIGNER_URL, signer_jar)

    # 提取 aapt2 二进制可执行文件
    if not aapt2_bin.exists() or aapt2_bin.stat().st_size == 0:
        with zipfile.ZipFile(aapt2_jar, "r") as zf:
            with zf.open("aapt2") as src, open(aapt2_bin, "wb") as dst:
                shutil.copyfileobj(src, dst)
        aapt2_bin.chmod(0o755)

    return aapt2_bin, ecj_jar, r8_jar, android_jar, signer_jar


def run_cmd(cmd: list[str], cwd: Path | None = None) -> None:
    res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"执行命令失败：{' '.join(str(c) for c in cmd)}")
        print("STDOUT:", res.stdout)
        print("STDERR:", res.stderr)
        sys.exit(res.returncode)


def build_apk(project_dir: Path, output_apk: Path) -> None:
    aapt2_bin, ecj_jar, r8_jar, android_jar, signer_jar = prepare_tools()
    build_dir = project_dir / "build"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    gen_dir = build_dir / "gen"
    classes_dir = build_dir / "classes"
    compiled_res = build_dir / "compiled_res.zip"
    unaligned_apk = build_dir / "unaligned.apk"
    gen_dir.mkdir(parents=True, exist_ok=True)
    classes_dir.mkdir(parents=True, exist_ok=True)

    print("1/5: 编译 Android 资源 (aapt2 compile)...")
    run_cmd([str(aapt2_bin), "compile", "--dir", str(project_dir / "res"), "-o", str(compiled_res)])

    print("2/5: 链接 Android 资源与清单 (aapt2 link)...")
    manifest = project_dir / "AndroidManifest.xml"
    assets_dir = project_dir / "assets"
    cmd = [
        str(aapt2_bin), "link",
        "-o", str(unaligned_apk),
        "-I", str(android_jar),
        "--manifest", str(manifest),
        "--java", str(gen_dir),
        "--auto-add-overlay",
    ]
    if assets_dir.exists():
        cmd.extend(["-A", str(assets_dir)])
    cmd.append(str(compiled_res))
    run_cmd(cmd)

    print("3/5: 编译 Java 源码 (ecj)...")
    java_files = [str(f) for f in (project_dir / "src").rglob("*.java")] + [str(f) for f in gen_dir.rglob("*.java")]
    run_cmd([
        "java", "-jar", str(ecj_jar),
        "-encoding", "UTF-8",
        "-1.8",
        "-cp", str(android_jar),
        "-d", str(classes_dir),
        *java_files,
    ])

    print("4/5: 生成 DEX 字节码 (d8/r8)...")
    class_files = [str(f) for f in classes_dir.rglob("*.class")]
    run_cmd([
        "java", "-cp", str(r8_jar),
        "com.android.tools.r8.D8",
        "--output", str(build_dir),
        "--lib", str(android_jar),
        "--min-api", "21",
        *class_files,
    ])

    # 将 classes.dex 打入 unaligned.apk
    dex_file = build_dir / "classes.dex"
    if not dex_file.exists():
        raise RuntimeError("未找到生成的 classes.dex")
    with zipfile.ZipFile(unaligned_apk, "a") as zf:
        zf.write(dex_file, "classes.dex")

    print("5/5: 签名与 Zipalign 对齐 (uber-apk-signer)...")
    signed_dir = build_dir / "signed"
    signed_dir.mkdir(parents=True, exist_ok=True)
    run_cmd([
        "java", "-jar", str(signer_jar),
        "--apks", str(unaligned_apk),
        "--out", str(signed_dir),
    ])

    # 寻找生成的最终 apk
    candidates = list(signed_dir.glob("*.apk"))
    if not candidates:
        raise RuntimeError("签名未产出 APK 文件")
    final_signed_apk = candidates[0]

    output_apk.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(final_signed_apk, output_apk)

    print(f"🎉 成功构建 Android APK 发行包：{output_apk} ({output_apk.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    proj = Path(__file__).resolve().parent
    out = proj.parent.parent / "dist" / "FutureWork-0.1.0-android.apk"
    build_apk(proj, out)
