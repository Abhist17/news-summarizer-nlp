# Windows equivalent of the Makefile.   Usage:  .\run.ps1 setup | train | eval | demo-cache | app | test | report | all
# If scripts are blocked:  Set-ExecutionPolicy -Scope Process Bypass
param([Parameter(Position = 0)][string]$Task = "help",
      [int]$TrainN = 3000, [int]$ValN = 300, [int]$TestN = 200, [int]$AbsN = 100)

$ErrorActionPreference = "Stop"
$Py = ".\.venv\Scripts\python.exe"

function Invoke-Step([string]$exe, [string[]]$argv) {
    & $exe @argv
    if ($LASTEXITCODE -ne 0) { throw "Command failed: $exe $($argv -join ' ')" }
}

switch ($Task) {
    "setup" {
        $sys = if (Get-Command py -ErrorAction SilentlyContinue) { "py" } else { "python" }
        if ($sys -eq "py") { Invoke-Step py @("-3.12", "-m", "venv", ".venv") } else { Invoke-Step python @("-m", "venv", ".venv") }
        Invoke-Step $Py @("-m", "pip", "install", "--upgrade", "pip")
        Invoke-Step $Py @("-m", "pip", "install", "-r", "requirements.txt")
        Invoke-Step $Py @("scripts/download_models.py")
    }
    "train"      { Invoke-Step $Py @("-m", "src.train_extractive", "--train-n", $TrainN, "--val-n", $ValN) }
    "eval"       { Invoke-Step $Py @("-m", "src.evaluate", "--n", $TestN, "--abs-n", $AbsN) }
    "demo-cache" { Invoke-Step $Py @("-m", "src.precompute_demo") }
    "app"        { Invoke-Step $Py @("-m", "streamlit", "run", "app.py") }
    "test"       { Invoke-Step $Py @("-m", "pytest", "-q") }
    "report"     { Invoke-Step $Py @("-m", "src.make_figures"); Invoke-Step $Py @("report/build_pdf.py") }
    "all" {
        Invoke-Step $Py @("-m", "src.train_extractive", "--train-n", $TrainN, "--val-n", $ValN)
        Invoke-Step $Py @("-m", "src.evaluate", "--n", $TestN, "--abs-n", $AbsN)
        Invoke-Step $Py @("-m", "src.precompute_demo")
        Invoke-Step $Py @("-m", "pytest", "-q")
    }
    default { Write-Host "Usage: .\run.ps1 setup | train | eval | demo-cache | app | test | report | all" }
}
