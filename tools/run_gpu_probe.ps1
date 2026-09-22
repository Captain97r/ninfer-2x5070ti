[CmdletBinding()]
param(
    [string]$CudaPath = $env:CUDA_PATH,
    [string]$OutputDirectory = (Join-Path $PSScriptRoot '..\diagnostics')
)
$ErrorActionPreference = 'Stop'
if (-not $CudaPath) { $CudaPath = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3' }
$workspace = Split-Path -Parent $PSScriptRoot
$outputPath = [IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Force -Path $outputPath | Out-Null
$buildPath = Join-Path $outputPath 'build'
New-Item -ItemType Directory -Force -Path $buildPath | Out-Null
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$vsPath = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vsPath) { throw 'Visual Studio C++ x64 Build Tools are required.' }
$vcvars = Join-Path $vsPath 'VC\Auxiliary\Build\vcvars64.bat'
$source = Join-Path $PSScriptRoot 'gpu_probe.cpp'
$exePath = Join-Path $buildPath 'gpu_probe.exe'
$objectPath = Join-Path $buildPath 'gpu_probe.obj'
# All values are local paths passed as quoted cmd arguments; no filesystem deletion.
$buildCommand = 'call "{0}" >nul && cl.exe /nologo /EHsc /O2 /std:c++17 /I"{1}\include" "{2}" /Fo"{3}" /Fe"{4}" /link /LIBPATH:"{1}\lib\x64" cudart.lib' -f $vcvars, $CudaPath, $source, $objectPath, $exePath
& $env:ComSpec /d /s /c $buildCommand
if ($LASTEXITCODE -ne 0) { throw "C++ compilation failed with exit code $LASTEXITCODE." }
$env:PATH = (Join-Path $CudaPath 'bin') + ';' + $env:PATH
$query = 'index,name,pci.bus_id,memory.total,memory.free,driver_version,pcie.link.gen.current,pcie.link.width.current,pcie.link.gen.max,pcie.link.width.max'
$idleStatus = @(& nvidia-smi "--query-gpu=$query" '--format=csv,noheader,nounits')
$started = [DateTime]::UtcNow.ToString('o')
$probeJsonPath = Join-Path $outputPath 'gpu_probe.json'
$probeLogPath = Join-Path $outputPath 'gpu_probe.stderr.log'
$process = Start-Process -FilePath $exePath -WorkingDirectory $buildPath -WindowStyle Hidden -PassThru -RedirectStandardOutput $probeJsonPath -RedirectStandardError $probeLogPath
$null = $process.Handle
$samples = @()
while (-not $process.HasExited) {
    $sampleTime = [DateTime]::UtcNow.ToString('o')
    $rows = @(& nvidia-smi '--query-gpu=index,pcie.link.gen.current,pcie.link.width.current,utilization.gpu,utilization.memory' '--format=csv,noheader,nounits')
    $samples += [pscustomobject]@{utc = $sampleTime; gpu_rows = $rows}
    Start-Sleep -Milliseconds 200
    $process.Refresh()
}
$process.WaitForExit()
if ($process.ExitCode -ne 0) {
    Get-Content -LiteralPath $probeLogPath
    throw "GPU probe failed with exit code $($process.ExitCode)."
}
$report = [ordered]@{
    schema_version = 1
    started_utc = $started
    completed_utc = [DateTime]::UtcNow.ToString('o')
    operating_system = Get-CimInstance Win32_OperatingSystem | Select-Object Caption, Version, BuildNumber
    cpu = @(Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors, MaxClockSpeed)
    motherboard = @(Get-CimInstance Win32_BaseBoard | Select-Object Manufacturer, Product, Version)
    physical_memory = @(Get-CimInstance Win32_PhysicalMemory | Select-Object Manufacturer, PartNumber, Capacity, Speed, ConfiguredClockSpeed)
    total_physical_memory_bytes = (Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory
    nvidia_smi_idle_columns = $query
    nvidia_smi_idle_rows = $idleStatus
    nvidia_smi_sample_columns = 'index,pcie.link.gen.current,pcie.link.width.current,utilization.gpu,utilization.memory'
    pcie_samples_during_probe = $samples
    probe = Get-Content -LiteralPath $probeJsonPath -Raw | ConvertFrom-Json
}
$reportPath = Join-Path $outputPath 'hardware_report.json'
$report | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Output "Hardware report: $reportPath"
Write-Output "Raw probe: $probeJsonPath"
