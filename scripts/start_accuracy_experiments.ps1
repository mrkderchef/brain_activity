param(
    [string]$RootDir = "outputs\accuracy_marathon",
    [switch]$Force,
    [switch]$SkipViterbi
)

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectDir

$arguments = @(
    "-u",
    "scripts\run_accuracy_experiments.py",
    "--root-dir",
    $RootDir
)
if ($Force) {
    $arguments += "--force"
}
if ($SkipViterbi) {
    $arguments += "--skip-viterbi"
}

$logDir = Join-Path $ProjectDir $RootDir
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stdout = Join-Path $logDir "marathon.stdout.log"
$stderr = Join-Path $logDir "marathon.stderr.log"

$process = Start-Process `
    -FilePath "python" `
    -ArgumentList $arguments `
    -WorkingDirectory $ProjectDir `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -WindowStyle Hidden `
    -PassThru

$pidPath = Join-Path $logDir "marathon.pid.txt"
Set-Content -Path $pidPath -Value $process.Id
Write-Host "Started accuracy marathon PID=$($process.Id)"
Write-Host "Stdout: $stdout"
Write-Host "Stderr: $stderr"
Write-Host "Summary will be written to: $(Join-Path $logDir 'accuracy_experiment_summary.md')"
