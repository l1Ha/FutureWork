# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['packaging/entrypoint.py'],
    pathex=['/media/lh/新加卷/aecho/AIcreating/FutureWork'],
    binaries=[],
    datas=[('/media/lh/新加卷/aecho/AIcreating/FutureWork/futurework/web/static', 'futurework/web/static'), ('/media/lh/新加卷/aecho/AIcreating/FutureWork/futurework/sensory/models/face_detection_yunet_2023mar.onnx', 'futurework/sensory/models')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='futurework',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
