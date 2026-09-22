[CmdletBinding()]
param(
    [string]$ModelPath,
    [ValidateSet('Cli', 'Server')][string]$Mode = 'Cli',
    [string]$Prompt = 'Explain how tensor parallel inference works in three sentences.',
    [string]$MessagesFile,
    [ValidateRange(512, 262144)][int]$Context = 199680,
    [ValidateRange(1, 32768)][int]$MaxNew = 256,
    [ValidateRange(0, 5)][int]$DraftTokens = 3,
    [ValidateRange(1, 65535)][int]$Port = 8000,
    [ValidateRange(1, 16384)][int]$ImageMaxTokens = 2048,
    [switch]$NoVision,
    [switch]$NoCudaGraph,
    [switch]$Greedy,
    [switch]$NoThinking,
    [switch]$IgnoreEos
)
$ErrorActionPreference = 'Stop'
$workspace = Split-Path -Parent $PSScriptRoot
$build = Join-Path $workspace 'build\windows'
if (-not $ModelPath) {
    $modelSpec = Get-Content -LiteralPath (Join-Path $workspace 'config\model.json') -Raw | ConvertFrom-Json
    $ModelPath = Join-Path $modelSpec.default_directory $modelSpec.filename
}
$binary = Join-Path $build $(if ($Mode -eq 'Server') { 'apps\ninfer-serve.exe' } else { 'apps\ninfer.exe' })
if (-not (Test-Path -LiteralPath $binary)) { throw 'Build the engine with tools/build_windows.ps1 first.' }
if (-not (Test-Path -LiteralPath $ModelPath -PathType Leaf)) { throw 'Model missing. Run tools/download_model.ps1 or pass -ModelPath.' }
$stream = [IO.File]::OpenRead($ModelPath)
try {
    $header = New-Object byte[] 8
    if ($stream.Read($header, 0, 8) -ne 8 -or [BitConverter]::ToString($header).Replace('-', '') -ne '4E494E4645520002') {
        throw 'This engine requires a v2 .ninfer artifact. GGUF and NInfer v3 files are incompatible.'
    }
} finally { $stream.Dispose() }
$env:PATH = (Join-Path $build 'vcpkg_installed\x64-windows\bin') + ';' + $env:PATH
$launchArgs = @($ModelPath, '--tp', '2', '--devices', '0,1', '--max-context', "$Context",
    '--kv-capacity', "$Context", '--kv-dtype', 'int8', '--prefill-chunk', '1024')
if ($DraftTokens -gt 0) { $launchArgs += @('--spec', 'mtp', '--draft-tokens', "$DraftTokens", '--lm-head-draft') }
if (-not $NoVision) { $launchArgs += @('--vision', '--image-max-tokens', "$ImageMaxTokens") }
if ($NoCudaGraph) { $launchArgs += '--no-cuda-graph' }
if ($NoThinking) { $launchArgs += '--no-thinking' }
if ($Mode -eq 'Server') {
    # OMP treats output truncation as a reason to compact. When the client omits
    # its output limit, let Engine clamp generation to the remaining context.
    $launchArgs += @('--host', '127.0.0.1', '--port', "$Port", '--max-concurrency', '1',
        '--default-max-tokens', "$Context")
} else {
    $launchArgs += @('--max-new', "$MaxNew")
    if ($MessagesFile) { $launchArgs += @('--messages', $MessagesFile) }
    else { $launchArgs += @('--prompt', $Prompt) }
    if ($Greedy) { $launchArgs += '--greedy' }
    if ($IgnoreEos) { $launchArgs += '--ignore-eos' }
}
$ErrorActionPreference = 'Continue' # Native stderr contains normal load/timing diagnostics.
& $binary @launchArgs
if ($LASTEXITCODE -ne 0) { throw "NInfer exited with code $LASTEXITCODE." }
