@echo off
rem Phase 3A-4/3A-5: Vision TP2 functional + memory verification on 2x RTX 5060 Ti.
rem Requires: GPU1 free (~13.4 GiB must be available for rank-1 weights + KV).
rem Usage: scripts\phase3a-vision-verify.cmd [artifact]
setlocal
set "ARTIFACT=%~1"
if "%ARTIFACT%"=="" set "ARTIFACT=M:\qwen\Qwen3.8-27b-nvfp4.ninfer"
set "ROOT=%~dp0.."
set "OUT=%ROOT%\logs\phase3a"
if not exist "%OUT%" mkdir "%OUT%"

echo === Vision TP2 verification run %date% %time% ===
echo artifact: %ARTIFACT%
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader > "%OUT%\gpu-before.txt"
type "%OUT%\gpu-before.txt"

set "NINFER=%ROOT%\..\build-windows-131\apps\ninfer.exe"
if not exist "%NINFER%" (
  echo ERROR: %NINFER% not found - build first
  exit /b 2
)

rem 1) image_chart.png - chart/diagram question
"%NINFER%" "%ARTIFACT%" --tp 2 --devices 0,1 --vision --greedy --max-new 128 ^
  --messages "%ROOT%\examples\cli\messages\image_chart.json" ^
  > "%OUT%\tp2_image_chart.out.txt" 2> "%OUT%\tp2_image_chart.err.txt"
echo image_chart exit=%ERRORLEVEL%

rem 2) natural_scene.png - photo description
"%NINFER%" "%ARTIFACT%" --tp 2 --devices 0,1 --vision --greedy --max-new 128 ^
  --messages "%ROOT%\examples\cli\messages\image_natural.json" ^
  > "%OUT%\tp2_image_natural.out.txt" 2> "%OUT%\tp2_image_natural.err.txt"
echo image_natural exit=%ERRORLEVEL%

rem 3) memory snapshot after the heaviest run
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader > "%OUT%\gpu-after.txt"
type "%OUT%\gpu-after.txt"

echo === done; outputs in %OUT% ===
endlocal
