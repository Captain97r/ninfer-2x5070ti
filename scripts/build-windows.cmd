@echo off
:: Native Windows build for this fork (branch windows-tp2).
:: Toolchain: Visual Studio 2022 MSVC x64 + CUDA 13.x + vcpkg manifest deps (FFmpeg,
::            libcurl, zlib) + Ninja generator.
::
:: Usage: cmd /c scripts\build-windows.cmd [configure^|build^|all]
::
:: Environment overrides (all optional):
::   NINFER_CUDA_BIN     nvcc.exe directory (default: CUDA found on PATH)
::   NINFER_VCPKG_ROOT   vcpkg checkout (default: third_party\vcpkg next to this script)
::   NINFER_VS_YEAR      Visual Studio edition to probe with vswhere (default: 2022)

setlocal
set "SRC=%~dp0.."
set "BUILD=%SRC%\build-windows"
set "VCPKG_ROOT=%NINFER_VCPKG_ROOT%"
if not defined VCPKG_ROOT set "VCPKG_ROOT=%SRC%\third_party\vcpkg"
if not defined NINFER_VS_YEAR set "NINFER_VS_YEAR=2022"

for /f "usebackq tokens=*" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -version ^[%NINFER_VS_YEAR%^] -property installationPath`) do set "VSPATH=%%i"
if not defined VSPATH ( echo vswhere could not locate Visual Studio %NINFER_VS_YEAR% & exit /b 1 )
call "%VSPATH%\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 ( echo vcvars64 failed & exit /b 1 )

set "ACTION=%~1"
if "%ACTION%"=="" set "ACTION=all"

if /i "%ACTION%"=="configure" goto :configure
if /i "%ACTION%"=="build"     goto :build
if /i "%ACTION%"=="all"       goto :configure

:configure
echo === CMake configure (Ninja, vcpkg manifest deps) ===
set "CMAKE_ARGS=-S "%SRC%" -B "%BUILD%" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE="%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake" -DVCPKG_TARGET_TRIPLET=x64-windows -DVCPKG_INSTALLED_DIR="%SRC%\build-windows\vcpkg_installed""
if defined NINFER_CUDA_BIN set "CMAKE_ARGS=%CMAKE_ARGS% -DCMAKE_CUDA_COMPILER="%NINFER_CUDA_BIN%\nvcc.exe""
cmake %CMAKE_ARGS%
if errorlevel 1 ( echo configure failed & exit /b 1 )
if /i "%ACTION%"=="configure" goto :eof

:build
echo === Build (Ninja, parallel) ===
cmake --build "%BUILD%" --parallel
if errorlevel 1 ( echo build failed & exit /b 1 )
echo === Build artifacts ===
dir /b "%BUILD%\apps\*.exe"
goto :eof
