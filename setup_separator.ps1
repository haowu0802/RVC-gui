#Requires -Version 5.1
<#
.SYNOPSIS
  Create or refresh the local audio-separator venv used by the Separate tab.
#>
param(
    [string]$Python = "python",
    [string]$TorchIndex = "https://download.pytorch.org/whl/cu128"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Venv = Join-Path $Root ".venv"
$Py = Join-Path $Venv "Scripts\python.exe"
$Pip = Join-Path $Venv "Scripts\pip.exe"
$Models = Join-Path $Root "models"

Write-Host "Root: $Root"
New-Item -ItemType Directory -Force -Path $Models | Out-Null

if (-not (Test-Path $Py)) {
    Write-Host "Creating venv at $Venv ..."
    & $Python -m venv $Venv
    if ($LASTEXITCODE -ne 0) { throw "venv create failed" }
}

& $Py -m pip install -U pip setuptools wheel
if ($LASTEXITCODE -ne 0) { throw "pip bootstrap failed" }

Write-Host "Installing torch from $TorchIndex ..."
& $Pip install torch torchaudio --index-url $TorchIndex
if ($LASTEXITCODE -ne 0) { throw "torch install failed" }

Write-Host "Installing audio-separator[gpu] ..."
& $Pip install "audio-separator[gpu]"
if ($LASTEXITCODE -ne 0) { throw "audio-separator install failed" }

$Exe = Join-Path $Venv "Scripts\audio-separator.exe"
if (-not (Test-Path $Exe)) { throw "audio-separator.exe missing after install: $Exe" }

Write-Host ""
Write-Host "OK: $Exe"
& $Py -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
Write-Host "Place MelBand/BS-RoFormer .ckpt files under: $Models"
Write-Host "Then launch the GUI and open the Separate tab."
