param(
    [switch]$CheckOnly,
    [switch]$SkipContainers,
    [ValidateRange(1024, 65535)][int]$Port = 8000
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Missing project .venv; follow root README.md.' }
if (-not (Test-Path -LiteralPath '.env')) { throw 'Missing root .env; configure it privately using .env.example.' }
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { throw 'Docker CLI is required.' }
& docker compose --env-file .env -f deploy/docker/compose.yaml config --quiet
if ($LASTEXITCODE -ne 0) { throw 'Compose configuration validation failed; values withheld.' }
& $pythonPath -B -m scripts.check_startup
if ($LASTEXITCODE -ne 0) { throw 'Python/config/JWT prerequisites failed; inspect key names in the runbook, never print .env.' }
if ($CheckOnly) { Write-Output '[OK] start prerequisites; no services started'; return }
if (-not $SkipContainers) {
    & docker compose --env-file .env -f deploy/docker/compose.yaml up -d --wait --wait-timeout 180
    if ($LASTEXITCODE -ne 0) { throw 'Middleware is not healthy; application was not started.' }
}
Write-Output "Open http://127.0.0.1:$Port ; Ctrl+C stops this application. Middleware volumes are retained."
& $pythonPath -m uvicorn app.main:app --host 127.0.0.1 --port $Port
if ($LASTEXITCODE -ne 0) { throw 'Application exited with an error; inspect sanitized diagnostics.' }
