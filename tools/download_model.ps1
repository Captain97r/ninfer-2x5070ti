[CmdletBinding()]
param(
    [string]$ManifestPath = (Join-Path $PSScriptRoot '..\config\model.json'),
    [string]$ModelDirectory,
    [switch]$ValidateOnly
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$modelSpec = Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json
if ($modelSpec.schema_version -ne 1 -or $modelSpec.container_version -ne 2) {
    throw 'The downloader requires a schema-1 manifest for a NInfer v2 artifact.'
}
if ($modelSpec.filename -ne [IO.Path]::GetFileName($modelSpec.filename) -or
    $modelSpec.sha256 -notmatch '^[0-9a-f]{64}$' -or
    $modelSpec.magic_hex -ne '4E494E4645520002' -or [long]$modelSpec.bytes -lt 16) {
    throw 'Invalid filename, size, checksum, or artifact magic in model manifest.'
}
if (-not $ModelDirectory) { $ModelDirectory = $modelSpec.default_directory }
$modelDirectoryPath = [IO.Path]::GetFullPath($ModelDirectory)
$modelPath = Join-Path $modelDirectoryPath $modelSpec.filename
$partialPath = $modelPath + '.part'

function Assert-ModelArtifact([string]$ArtifactPath) {
    $artifactInfo = Get-Item -LiteralPath $ArtifactPath
    if ($artifactInfo.Length -ne [long]$modelSpec.bytes) {
        throw "Size mismatch: $ArtifactPath has $($artifactInfo.Length) bytes; expected $($modelSpec.bytes). File preserved."
    }
    $artifactStream = [IO.File]::OpenRead($ArtifactPath)
    try {
        $prefix = New-Object byte[] 16
        if ($artifactStream.Read($prefix, 0, $prefix.Length) -ne $prefix.Length) {
            throw "Cannot read the artifact prefix: $ArtifactPath"
        }
        $actualMagic = [BitConverter]::ToString($prefix, 0, 8).Replace('-', '')
        if ($actualMagic -ne $modelSpec.magic_hex) {
            throw "Artifact magic mismatch: $ArtifactPath is not NInfer v2. File preserved."
        }
        $jsonBytes = [BitConverter]::ToUInt64($prefix, 8)
        if ($jsonBytes -eq 0 -or $jsonBytes -gt ([long]$modelSpec.bytes - 16)) {
            throw "Invalid JSON directory length in $ArtifactPath. File preserved."
        }
    } finally {
        $artifactStream.Dispose()
    }
    Write-Host "Verifying SHA-256: $ArtifactPath"
    $actualHash = (Get-FileHash -LiteralPath $ArtifactPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualHash -ne $modelSpec.sha256) {
        throw "SHA-256 mismatch: $ArtifactPath. Expected $($modelSpec.sha256), received $actualHash. File preserved."
    }
}

if ($ValidateOnly) {
    Assert-ModelArtifact $modelPath
    Write-Output "Verified model: $modelPath"
    return
}

New-Item -ItemType Directory -Force -Path $modelDirectoryPath | Out-Null
# Only one instance of this downloader may use this destination at a time.
$downloadLock = [IO.File]::Open($modelPath + '.download.lock', [IO.FileMode]::OpenOrCreate,
    [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
try {
    if (Test-Path -LiteralPath $modelPath) {
        Assert-ModelArtifact $modelPath
        Write-Output "Existing model verified: $modelPath"
        return
    }
    $partialBytes = 0L
    if (Test-Path -LiteralPath $partialPath) {
        $partialBytes = (Get-Item -LiteralPath $partialPath).Length
        if ($partialBytes -gt [long]$modelSpec.bytes) {
            throw "Partial file is larger than the expected artifact: $partialPath. File preserved."
        }
    }
    if ($partialBytes -lt [long]$modelSpec.bytes) {
        $volume = [IO.DriveInfo]::new([IO.Path]::GetPathRoot($modelDirectoryPath))
        $requiredBytes = [long]$modelSpec.bytes - $partialBytes + 256MB
        if ($volume.AvailableFreeSpace -lt $requiredBytes) {
            throw "Insufficient free disk space: need $requiredBytes bytes to finish the download."
        }
        $curlPath = (Get-Command curl.exe -CommandType Application -ErrorAction Stop).Source
        Write-Host "Downloading pinned revision $($modelSpec.revision) from byte $partialBytes."
        Write-Host "Partial file: $partialPath"
        & $curlPath --location --fail --retry 5 --retry-delay 2 --connect-timeout 30 `
            --continue-at - --output $partialPath --url $modelSpec.url
        if ($LASTEXITCODE -ne 0) {
            throw "curl failed with exit code $LASTEXITCODE. Partial file retained; rerun to resume: $partialPath"
        }
    }
    Assert-ModelArtifact $partialPath
    # File.Move has no overwrite behavior: an existing destination is never replaced.
    [IO.File]::Move($partialPath, $modelPath)
    Write-Output "Downloaded and verified model: $modelPath"
} finally {
    $downloadLock.Dispose()
}
