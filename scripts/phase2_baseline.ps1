# Phase 2: baseline check. Runs the tests, starts the app, calls the API, stops the app,
# and writes everything to phase2_output.txt in the project folder.
#   powershell -ExecutionPolicy Bypass -File scripts\phase2_baseline.ps1
param([string]$Python = '')
$ErrorActionPreference = 'Continue'
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
# Prefer the Python 3.10 environment from setup.bat; fall back to the existing Python 3.13 one.
$py = Join-Path $root '.venv\Scripts\python.exe'
if ($Python) { $py = $Python }
if (-not (Test-Path $py)) { $py = Join-Path $root '.venv313\Scripts\python.exe' }
if (-not (Test-Path $py)) { Write-Host 'No environment found. Run setup.bat first.'; exit 1 }
Write-Host "Using $py"
& $py -c "import pytest" 2>$null
if ($LASTEXITCODE -ne 0) { Write-Host 'installing pytest'; & $py -m pip install pytest==8.3.3 | Out-Null }
$env:PYTHONIOENCODING = 'utf-8'
$log = Join-Path $root 'phase2_output.txt'
'' | Out-File -FilePath $log -Encoding utf8
function Log($text) { $text | Out-File -FilePath $log -Append -Encoding utf8 }
function Clip($text, $n) { if ($text.Length -gt $n) { $text.Substring(0, $n) + ' ...[cut]' } else { $text } }

Log "=== Phase 2 baseline $(Get-Date -Format s) ==="
Log "interpreter: $py"
Log '--- python and library versions ---'
& $py -c "import sys,platform;print(sys.version);print(platform.platform())" 2>&1 | ForEach-Object { Log $_ }
foreach ($m in 'numpy', 'scipy', 'sklearn', 'skimage', 'cv2', 'torch', 'torchvision', 'tensorflow', 'flask', 'joblib', 'h5py') {
    Write-Host "checking $m"
    & $py -c "import $m;print('$m', getattr($m,'__version__','?'))" 2>&1 | Select-Object -Last 1 | ForEach-Object { Log $_ }
}

Write-Host 'running pytest (several minutes)'
Log '--- pytest -q ---'
& $py -m pytest -q -p no:cacheprovider 2>&1 | ForEach-Object { Log $_ }

Write-Host 'starting the app'
Log '--- app ---'
$stdout = Join-Path $root 'phase2_app_stdout.txt'
$stderr = Join-Path $root 'phase2_app_stderr.txt'
$proc = Start-Process -FilePath $py -ArgumentList 'app.py' -WorkingDirectory $root `
    -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
$base = 'http://127.0.0.1:5001'
$up = $false
for ($i = 0; $i -lt 90; $i++) {
    Start-Sleep -Seconds 2
    if ($proc.HasExited) { break }
    try { Invoke-WebRequest "$base/api/health" -UseBasicParsing -TimeoutSec 3 | Out-Null; $up = $true; break }
    catch { if ($_.Exception.Response) { $up = $true; break } }   # 503 means up but degraded
}
Log "server reachable: $up   process exited early: $($proc.HasExited)"

function Get-Url($url) {
    try { $r = Invoke-WebRequest $url -UseBasicParsing; return @{ code = [int]$r.StatusCode; body = $r.Content } }
    catch {
        $resp = $_.Exception.Response
        if ($resp) {
            $sr = New-Object IO.StreamReader($resp.GetResponseStream())
            return @{ code = [int]$resp.StatusCode; body = $sr.ReadToEnd() }
        }
        return @{ code = 0; body = $_.Exception.Message }
    }
}

if ($up) {
    $page = Get-Url "$base/"
    Log "--- GET / --- HTTP $($page.code), $($page.body.Length) bytes"
    $h = Get-Url "$base/api/health"
    Log "--- GET /api/health --- HTTP $($h.code)"; Log (Clip $h.body 2000)
    $i = Get-Url "$base/api/info"
    Log "--- GET /api/info --- HTTP $($i.code)"; Log (Clip $i.body 7000)
    $m = Get-Url "$base/api/metrics"
    Log "--- GET /api/metrics --- HTTP $($m.code)"
    try {
        $j = $m.body | ConvertFrom-Json
        Log "evaluation_available=$($j.evaluation_available)  hienwrite_evaluation_available=$($j.hienwrite.evaluation_available)  legacy_recorded=$([bool]$j.legacy_recorded)"
    } catch { Log (Clip $m.body 800) }
    foreach ($img in '1.1.jpg', '1.2.jpg', '3.1.jpg', '3.3.jpg') {
        if (-not (Test-Path (Join-Path $root $img))) { Log "--- POST $img --- file not found (moved?)"; continue }
        Write-Host "predicting $img"
        $res = & curl.exe -s -w "`nHTTP %{http_code}" -F "file=@$img" "$base/api/predict" 2>&1 | Out-String
        Log "--- POST /api/predict $img ---"; Log (Clip $res 2500)
    }
}
if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
Log '--- server log (stderr, last 80 lines) ---'
if (Test-Path $stderr) { Get-Content $stderr -Tail 80 | ForEach-Object { Log $_ } }
Log '--- server log (stdout, last 20 lines) ---'
if (Test-Path $stdout) { Get-Content $stdout -Tail 20 | ForEach-Object { Log $_ } }
Remove-Item $stdout, $stderr -ErrorAction SilentlyContinue
Write-Host ''
Write-Host "Done. Send me the contents of $log"
