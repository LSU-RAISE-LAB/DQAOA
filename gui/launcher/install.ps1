$ErrorActionPreference = 'Stop'
$AppRoot = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $AppRoot
# Missing Python versions are an expected detection result, not a setup error.
# Capture native stderr separately so PowerShell's Stop preference cannot turn
# a launcher's diagnostic into a terminating NativeCommandError.
function Probe-Python312([string]$Executable, [string]$Prefix = '') {
    $process = $null
    try {
        $info = New-Object System.Diagnostics.ProcessStartInfo
        $info.FileName = $Executable
        $info.Arguments = $Prefix + ' -c "import sys; print(sys.executable); sys.exit(0 if sys.version_info[:2] == (3,12) else 1)"'
        $info.UseShellExecute = $false
        $info.CreateNoWindow = $true
        $info.RedirectStandardOutput = $true
        $info.RedirectStandardError = $true
        $process = New-Object System.Diagnostics.Process
        $process.StartInfo = $info
        [void]$process.Start()
        $stdoutTask = $process.StandardOutput.ReadToEndAsync()
        $stderrTask = $process.StandardError.ReadToEndAsync()
        if (-not $process.WaitForExit(30000)) {
            $process.Kill()
            $process.WaitForExit()
            return $null
        }
        $output = $stdoutTask.Result.Trim()
        [void]$stderrTask.Result
        if ($process.ExitCode -eq 0 -and $output -and
            (Test-Path -LiteralPath $output -PathType Leaf)) {
            return $output
        }
    } catch {
        # An absent/broken candidate should not prevent trying the installer.
        return $null
    } finally {
        if ($null -ne $process) { $process.Dispose() }
    }
    return $null
}
function Find-Python312 {
    $candidates = @(
        "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
        "$env:ProgramFiles\Python312\python.exe"
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            $found = Probe-Python312 $candidate
            if ($found) { return $found }
        }
    }
    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($launcher) {
        $found = Probe-Python312 $launcher.Source '-3.12'
        if ($found) { return $found }
    }
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd -and $cmd.Source -notlike '*WindowsApps*') {
        $found = Probe-Python312 $cmd.Source
        if ($found) { return $found }
    }
    return $null
}
try {
    Write-Host 'Setting up DQAOA. Internet access is required.'
    $PythonExe = Find-Python312
    if (-not $PythonExe) {
        if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
            Start-Process 'https://www.python.org/downloads/windows/'
            throw 'Python 3.12 is needed. Install Python 3.12 (64-bit) from the opened page, then double-click Install Requirements again.'
        }
        Write-Host 'Installing Python 3.12 for your Windows account...'
        & winget install --id Python.Python.3.12 --exact --source winget --scope user --accept-source-agreements --accept-package-agreements
        if ($LASTEXITCODE -ne 0) { throw 'Python installation did not complete. Check the installer message above.' }
        $PythonExe = Find-Python312
        if (-not $PythonExe) { throw 'Python was installed but could not be located. Close this window and run Install Requirements again.' }
    }
    $VenvPython = Join-Path $AppRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $VenvPython)) {
        & $PythonExe -m venv (Join-Path $AppRoot '.venv')
        if ($LASTEXITCODE -ne 0) { throw 'Could not create the private app environment.' }
    }
    & $VenvPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw 'Could not update pip. Check your internet connection.' }
    & $VenvPython -m pip install -r (Join-Path $AppRoot 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed. See the message above.' }
    & $VenvPython -c 'import streamlit, qiskit, qiskit_aer, numpy, pandas, matplotlib'
    if ($LASTEXITCODE -ne 0) { throw 'Dependency verification failed.' }
    Set-Content -LiteralPath (Join-Path $AppRoot '.venv\dqaoa-ready.txt') -Value 'ready'
    Write-Host ''
    Write-Host 'Installation complete. Close this window and double-click Open App.cmd.' -ForegroundColor Green
    exit 0
} catch {
    Write-Host "Setup could not finish: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
