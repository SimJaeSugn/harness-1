# 아리아 RAG POC 실행 스크립트 (Windows PowerShell)
# 사용법:  .\poc\run.ps1
# 가상환경 활성화 후 의존성 설치 + 서버 기동.

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here

if (-not (Test-Path ".env")) {
  Copy-Item ".env.example" ".env"
  Write-Host ".env 생성됨 (.env.example 복사). 필요 시 값 수정 후 다시 실행하세요." -ForegroundColor Yellow
}

if (-not (Test-Path ".venv")) {
  Write-Host "가상환경 생성 중..." -ForegroundColor Cyan
  python -m venv .venv
}
& ".\.venv\Scripts\python.exe" -m pip install --quiet --upgrade pip
& ".\.venv\Scripts\python.exe" -m pip install --quiet -r requirements.txt

Write-Host "서버 기동: http://127.0.0.1:8080" -ForegroundColor Green
& ".\.venv\Scripts\python.exe" -m uvicorn backend.app:app --host 0.0.0.0 --port 8080
