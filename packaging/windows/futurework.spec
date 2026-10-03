# -*- mode: python ; coding: utf-8 -*-

import os
import sys

block_cipher = None
project_root = os.path.abspath(os.path.join(SPECPATH, '..', '..'))

a = Analysis(
    [os.path.join(project_root, 'packaging', 'entrypoint.py')],
    pathex=[project_root],
    binaries=[],
    datas=[
        (os.path.join(project_root, 'futurework', 'web', 'static'), 'futurework/web/static'),
    ],
    hiddenimports=[
        'futurework',
        'futurework.types',
        'futurework.sensory',
        'futurework.cognition',
        'futurework.tools',
        'futurework.resilience',
        'futurework.runtime',
        'futurework.web',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
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
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=os.path.join(project_root, 'packaging', 'icon.ico'),
)
