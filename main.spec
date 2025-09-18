# main.spec
from PyInstaller.utils.hooks import copy_metadata, collect_submodules

datas = []
# collect metadata for required libraries
for pkg in ["streamlit", "pandas", "numpy", "matplotlib", "plotly"]:
    try:
        datas += copy_metadata(pkg)
    except Exception:
        pass  # if metadata not found, ignore

hiddenimports = []
for pkg in ["streamlit", "pandas", "numpy", "matplotlib", "plotly"]:
    hiddenimports += collect_submodules(pkg)

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
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
    [],
    exclude_binaries=True,
    name='DrivingDashboard',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    onefile=True,
    console=True  # change to True for debugging
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='DrivingDashboard'
)
