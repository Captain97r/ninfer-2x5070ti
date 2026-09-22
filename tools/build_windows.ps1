[CmdletBinding()]
param(
    [ValidateSet('Configure', 'Build', 'All')][string]$Action = 'All',
    [string]$VcpkgRoot = $env:NINFER_VCPKG_ROOT,
    [string]$PythonExecutable,
    [string[]]$Target = @('ninfer', 'ninfer-serve'),
    [switch]$Tests,
    [switch]$Benchmarks,
    [switch]$Fresh
)
$ErrorActionPreference = 'Stop'
# Windows PowerShell 5.1 converts redirected native stderr to ErrorRecords, even for
# successful compiler warnings. Native process exit status is the success criterion.
function Invoke-BuildTool([string]$Command, [string[]]$CommandArguments) {
    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $Command @CommandArguments
        if ($LASTEXITCODE -ne 0) { throw "$Command failed with exit code $LASTEXITCODE." }
    } finally { $ErrorActionPreference = $previousPreference }
}
$workspace = Split-Path -Parent $PSScriptRoot
$source = $workspace
$build = Join-Path $workspace 'build\windows'
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path -LiteralPath $vswhere)) { throw 'Install Visual Studio C++ x64 Build Tools.' }
$vsPath = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vsPath) { throw 'Visual Studio C++ x64 Build Tools were not found.' }
# Import the compiler environment into this process; subsequent commands use argument arrays.
$vcvars = Join-Path $vsPath 'VC\Auxiliary\Build\vcvars64.bat'
$compilerEnvironment = & $env:ComSpec /d /s /c ('call "{0}" >nul && set' -f $vcvars)
if ($LASTEXITCODE -ne 0) { throw 'Could not initialize the MSVC x64 environment.' }
foreach ($line in $compilerEnvironment) {
    if ($line -match '^([^=]+)=(.*)$') { Set-Item -LiteralPath "Env:$($matches[1])" -Value $matches[2] }
}
# CMake and Ninja must decode localized /showIncludes bytes identically. VSLANG
# alone is insufficient when only a non-English MSVC language pack is installed.
& chcp.com 65001 > $null
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
$env:VSLANG = '1033'
if ($Action -ne 'Build') {
    if (-not $VcpkgRoot) { $VcpkgRoot = Join-Path $vsPath 'VC\vcpkg' }
    $toolchain = Join-Path $VcpkgRoot 'scripts\buildsystems\vcpkg.cmake'
    if (-not (Test-Path -LiteralPath $toolchain)) { throw 'Set -VcpkgRoot to a vcpkg installation.' }
    if (-not $PythonExecutable) {
        # Include uv-managed Python installations listed by the Windows launcher.
        foreach ($interpreter in @(& py -0p)) {
            if ($interpreter -match '3\.11\S*\s+\*?\s*([A-Za-z]:\\.*python\.exe)\s*$') {
                $PythonExecutable = $matches[1]
                break
            }
        }
        if (-not $PythonExecutable) { throw 'Python 3.11 is required for engine tooling; pass -PythonExecutable.' }
    }
    & $PythonExecutable -c 'import sys; assert sys.version_info[:2] == (3, 11), "Python 3.11 is required"'
    if ($LASTEXITCODE -ne 0) { throw 'Python interpreter validation failed.' }
    $nvcc = (Get-Command nvcc.exe -ErrorAction Stop).Source
    New-Item -ItemType Directory -Force -Path $build | Out-Null
    $env:VCPKG_DOWNLOADS = Join-Path $build 'downloads'
    New-Item -ItemType Directory -Force -Path $env:VCPKG_DOWNLOADS | Out-Null
    $configureArgs = @('-S', $source, '-B', $build, '-G', 'Ninja',
        '-DCMAKE_BUILD_TYPE=Release', "-DCMAKE_TOOLCHAIN_FILE=$toolchain",
        '-DVCPKG_TARGET_TRIPLET=x64-windows', "-DVCPKG_INSTALLED_DIR=$build/vcpkg_installed",
        "-DCMAKE_CUDA_COMPILER=$nvcc", '-DCMAKE_CUDA_ARCHITECTURES=120a',
        "-DPython3_EXECUTABLE=$PythonExecutable", '-DNINFER_BUILD_APPS=ON',
        ('-DBUILD_TESTING=' + $(if ($Tests) { 'ON' } else { 'OFF' })),
        ('-DNINFER_BUILD_BENCHMARKS=' + $(if ($Benchmarks) { 'ON' } else { 'OFF' })))
    if ($Fresh) { $configureArgs += '--fresh' }
    Invoke-BuildTool 'cmake' $configureArgs
}
if ($Action -ne 'Configure') {
    $buildArgs = @('--build', $build, '-j')
    if ($Target.Count) { $buildArgs += '--target'; $buildArgs += $Target }
    Invoke-BuildTool 'cmake' $buildArgs
}
