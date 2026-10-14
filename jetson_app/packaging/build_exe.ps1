# Windows용 jetson-studio.exe를 만든다(deviation 전용이라 torch를 넣지 않는다).
# 사용: 프로젝트(jetson_app) 폴더에서  powershell -ExecutionPolicy Bypass -File packaging/build_exe.ps1
$ErrorActionPreference = "Stop"
uv run --with "pyinstaller<6.10" pyinstaller `
    --noconfirm --clean --onedir --name jetson-studio `
    --paths src `
    --add-data "src/jetson_app/web;jetson_app/web" `
    --exclude-module torch `
    packaging/studio_entry.py
Write-Host "완료: dist/jetson-studio/jetson-studio.exe"
