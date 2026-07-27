$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VirtualEnvironment = Join-Path $ProjectRoot ".venv"
$Python = Join-Path $VirtualEnvironment "Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    python -m venv $VirtualEnvironment
}

& $Python -m pip install -r (Join-Path $ProjectRoot "requirements.txt")
& $Python -m unittest discover -s (Join-Path $ProjectRoot "tests") -v
Push-Location $ProjectRoot
try {
    & $Python -m src.train
}
finally {
    Pop-Location
}

