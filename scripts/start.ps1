$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

$venvPython = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
$venvPython = [System.IO.Path]::GetFullPath($venvPython)
$venvUv = Join-Path $PSScriptRoot "..\.venv\Scripts\uv.exe"
$venvUv = [System.IO.Path]::GetFullPath($venvUv)

if (-not (Test-Path $venvPython)) {
    $python313 = Get-Command py -ErrorAction SilentlyContinue
    if ($python313) {
        & py -3.13 -m venv .venv
    } else {
        throw "Python 3.13 is required. Install Python 3.13 and reopen PowerShell."
    }
}

if (-not (Test-Path $venvUv)) {
    & $venvPython -m pip install --upgrade uv
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to install uv in .venv."
    }
}

& $venvUv sync --locked --all-extras
if ($LASTEXITCODE -ne 0) {
    throw "Dependency sync failed."
}

& $venvUv run file-tagger
