# Phase 0: backup + cleanup of macOS leftovers.
# Run from the project folder in PowerShell:
#   powershell -ExecutionPolicy Bypass -File scripts\phase0_cleanup.ps1
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
$stamp = Get-Date -Format 'yyyyMMdd_HHmm'
$backup = Join-Path $env:USERPROFILE "Documents\HandParsona_backup_$stamp"

Write-Host "[1/4] Backing up to $backup (skips virtual environments and ._ files)"
robocopy $root $backup /E /XD .venv .venv310 .venv313 venv __pycache__ .pytest_cache /XF "._*" .DS_Store /NFL /NDL /NP | Out-Null
if ($LASTEXITCODE -ge 8) { throw "Backup failed (robocopy exit code $LASTEXITCODE)" }
$mustExist = @('best_model.h5', 'personality_prediction_resnet_model.h5', 'models\hienwrite\svr_hog.joblib',
               'models\hienwrite\random_forest_hog_lbp.joblib', 'models\hienwrite\resnet18_regression.pt')
foreach ($f in $mustExist) {
    if (-not (Test-Path (Join-Path $backup $f))) { throw "Backup is missing $f - stopping before any deletion." }
}
Write-Host "      Backup verified: model weights present."

Write-Host '[2/4] Marking the current git commit so it can be restored'
if (Get-Command git -ErrorAction SilentlyContinue) {
    git branch "backup-before-cleanup-$stamp" 2>$null
} else {
    Write-Host '      git not found, skipping (the folder backup above is still valid).'
}

Write-Host '[3/4] Deleting macOS metadata files (._* and .DS_Store)'
$junk = Get-ChildItem -Path $root -Recurse -Force -File -ErrorAction SilentlyContinue |
    Where-Object { ($_.Name -like '._*' -or $_.Name -eq '.DS_Store') -and $_.FullName -notmatch '\\\.git\\' }
$count = ($junk | Measure-Object).Count
$junk | Remove-Item -Force -ErrorAction SilentlyContinue
Write-Host "      Removed $count files."

Write-Host '[4/4] Removing macOS virtual environments and caches'
foreach ($d in '.venv310', '.venv313', 'venv', '.venv') {
    $p = Join-Path $root $d
    # macOS layout has bin\ and no Scripts\; a Windows venv has Scripts\ and is left alone.
    if ((Test-Path "$p\bin") -and -not (Test-Path "$p\Scripts")) {
        Remove-Item $p -Recurse -Force -ErrorAction SilentlyContinue
        Write-Host "      Removed macOS environment $d"
    }
}
Remove-Item (Join-Path $root '.pytest_cache') -Recurse -Force -ErrorAction SilentlyContinue
Get-ChildItem -Path $root -Recurse -Force -Directory -Filter '__pycache__' -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -notmatch '\\\.venv\\' } |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

Write-Host ''
Write-Host 'Phase 0 finished. Next: setup.bat, then scripts\phase2_baseline.ps1'
