# Windows용 jetson-studio.exe를 만든다(deviation 전용이라 torch를 넣지 않는다).
# 사용: 프로젝트(jetson_app) 폴더에서  powershell -ExecutionPolicy Bypass -File packaging/build_exe.ps1
# 실행 중인 exe가 있으면 dist 폴더를 지우지 못해 실패한다. 그때는 앱을 끄거나 -DistPath로 다른 폴더에 만든다.
param(
    [string]$DistPath = "dist",
    [string]$WorkPath = "build"
)
$ErrorActionPreference = "Stop"
uv run --with "pyinstaller<6.10" pyinstaller `
    --noconfirm --clean --onedir --name jetson-studio `
    --distpath $DistPath --workpath $WorkPath `
    --paths src `
    --add-data "src/jetson_app/web;jetson_app/web" `
    --exclude-module torch `
    packaging/studio_entry.py
Write-Host "완료: $DistPath/jetson-studio/jetson-studio.exe"
