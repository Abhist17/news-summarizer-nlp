# Windows equivalent of the Makefile.   Usage:  .\run.ps1 setup | models | train | eval | eval-full | demo-cache | app | test | report | all
# If scripts are blocked:  Set-ExecutionPolicy -Scope Process Bypass
# `setup` installs CUDA PyTorch if an NVIDIA GPU is found; add -Cpu to force the CPU-only wheel.
param([Parameter(Position = 0)][string]$Task = "help",
      [int]$TrainN = 3000, [int]$ValN = 300, [int]$TestN = 200, [int]$AbsN = 100, [switch]$Cpu)

$ErrorActionPreference = "Stop"
# The venv is several GB. In a OneDrive-synced folder (e.g. a OneDrive Desktop) syncing it is slow and locks
# files mid-install, so there it lives in %USERPROFILE%\.venvs\<project> instead. Override with $env:SUMM_VENV.
if ($env:SUMM_VENV) { $VenvDir = $env:SUMM_VENV }
elseif ($PSScriptRoot -match '\\OneDrive( - [^\\]+)?\\') { $VenvDir = Join-Path $env:USERPROFILE ".venvs\$(Split-Path $PSScriptRoot -Leaf)" }
else { $VenvDir = Join-Path $PSScriptRoot ".venv" }
$Py = Join-Path $VenvDir "Scripts\python.exe"
$env:PYTHONUTF8 = "1"   # UTF-8 for files and redirected output (Windows defaults to cp1252)

function Invoke-Step([string]$exe, [string[]]$argv) {
    & $exe @argv
    if ($LASTEXITCODE -ne 0) { throw "Command failed: $exe $($argv -join ' ')" }
}

switch ($Task) {
    "setup" {
        if (-not (Test-Path $Py)) {
            $sys = if (Get-Command py -ErrorAction SilentlyContinue) { "py" } else { "python" }
            if ($sys -eq "py") { Invoke-Step py @("-3.12", "-m", "venv", $VenvDir) } else { Invoke-Step python @("-m", "venv", $VenvDir) }
        }
        Write-Host "Python environment: $VenvDir"
        Invoke-Step $Py @("-m", "pip", "install", "--upgrade", "pip")
        # Plain `pip install torch` on Windows is CPU-only: pull the CUDA 12.4 wheel when an NVIDIA GPU is present.
        $wheel = if ($Cpu -or -not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) { "cpu" } else { "cu124" }
        Write-Host "Installing PyTorch ($wheel) ..."
        Invoke-Step $Py @("-m", "pip", "install", "torch==2.5.1", "--index-url", "https://download.pytorch.org/whl/$wheel")
        Invoke-Step $Py @("-m", "pip", "install", "-r", "requirements.txt")
        Invoke-Step $Py @("scripts/download_models.py")
        Invoke-Step $Py @("-c", "import torch; print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')")
    }
    "models"     { Invoke-Step $Py @("scripts/download_models.py") }
    "train"      { Invoke-Step $Py @("-m", "src.train_extractive", "--train-n", $TrainN, "--val-n", $ValN) }
    "eval"       { Invoke-Step $Py @("-m", "src.evaluate", "--n", $TestN, "--abs-n", $AbsN) }
    "eval-full"  { Invoke-Step $Py @("-m", "src.evaluate", "--n", "1000", "--abs-n", "1000", "--val-n", "300", "--xsum", "--xsum-n", "100") }
    "demo-cache" { Invoke-Step $Py @("-m", "src.precompute_demo") }
    "app"        { Invoke-Step $Py @("-m", "streamlit", "run", "app.py", "--server.headless=false") }
    "test"       { Invoke-Step $Py @("-m", "pytest", "-q") }
    "report"     { Invoke-Step $Py @("-m", "src.make_figures"); Invoke-Step $Py @("report/build_pdf.py") }
    "all" {
        Invoke-Step $Py @("-m", "src.train_extractive", "--train-n", $TrainN, "--val-n", $ValN)
        Invoke-Step $Py @("-m", "src.evaluate", "--n", $TestN, "--abs-n", $AbsN)
        Invoke-Step $Py @("-m", "src.precompute_demo")
        Invoke-Step $Py @("-m", "pytest", "-q")
    }
    default { Write-Host "Usage: .\run.ps1 setup | models | train | eval | eval-full | demo-cache | app | test | report | all" }
}
