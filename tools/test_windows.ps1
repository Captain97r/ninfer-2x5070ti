[CmdletBinding()]
param([switch]$Model)
$ErrorActionPreference = 'Stop'
$workspace = Split-Path -Parent $PSScriptRoot
$build = Join-Path $workspace 'build\windows'
$testNames = @('tensor_slice', 'kv_capacity_tp2', 'peer_mailbox', 'allreduce',
    'linear_nvfp4_a16', 'linear_nvfp4_a4', 'mtp_split', 'gdn_headsplit',
    'gqa_tp2_sm70', 'bench_support')
$targets = @($testNames | ForEach-Object { "ninfer_${_}_test" })
if ($Model) {
    $modelSpec = Get-Content -LiteralPath (Join-Path $workspace 'config\model.json') -Raw | ConvertFrom-Json
    $modelPath = Join-Path $modelSpec.default_directory $modelSpec.filename
    if (-not (Test-Path -LiteralPath $modelPath)) { throw 'Run tools/download_model.ps1 first.' }
    $env:NINFER_QWEN3_8_27B_WEIGHTS = $modelPath
    $targets += 'ninfer_qwen3_8_27b_prefix_tp2_real_test'
}
& (Join-Path $PSScriptRoot 'build_windows.ps1') -Action All -Tests -Benchmarks -Target $targets
$env:PATH = (Join-Path $build 'vcpkg_installed\x64-windows\bin') + ';' + $env:PATH
$filter = '^(' + (($targets | ForEach-Object { [regex]::Escape($_) }) -join '|') + ')$'
# Serial GPU checks; running kernels concurrently would invalidate timings and residency.
$ErrorActionPreference = 'Continue'
& ctest --test-dir $build --output-on-failure --no-tests=error -R $filter
if ($LASTEXITCODE -ne 0) { throw 'NInfer correctness checks failed.' }
