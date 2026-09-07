@echo off
:: Phase 3A-1 probe build: host-only tool, links the production libs.
setlocal
set "WS=D:\AI_envs\aider_olymp\project_ninfer"
set "SRC=%WS%\forks\ninfer-tp2-1m"
set "CUDA131=%WS%\third_party\cuda-13.1"
call "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 ( echo vcvars64 failed & exit /b 1 )
cd /d "%SRC%"
if not exist "%WS%\build\diagnostics" mkdir "%WS%\build\diagnostics"
cl /nologo /EHsc /O2 /std:c++20 /W3 /MD ^
  /I src /I include /I "%CUDA131%\include" /I src\targets\qwen3_6_27b\export ^
  /I src\targets\qwen3_6\export /I src\targets\qwen3_6_27b ^
  tools\tp2\vision_ws_probe.cpp /Fe"%WS%\build\diagnostics\vision_ws_probe.exe" ^
  /Fo"%WS%\build\diagnostics\vision_ws_probe.obj" ^
  /link /LIBPATH:"%CUDA131%\lib\x64" cudart_static.lib cuda.lib ^
  /LIBPATH:"%WS%\build-windows-131\src" ninfer_ops.lib ninfer_core.lib ninfer_nvfp4_tma.lib
if errorlevel 1 ( echo build failed & exit /b 1 )
endlocal
