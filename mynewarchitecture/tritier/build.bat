@echo off
setlocal enabledelayedexpansion

echo ===============================================================================
echo PROJECT B -- TRITIER: FULL SYSTEM BUILD, VALIDATION, AND BENCHMARK SUITE
echo ===============================================================================

REM 1. Detect MSVC Environment
if not defined DevEnvDir (
    if exist "D:\vs studio\VC\Auxiliary\Build\vcvarsall.bat" (
        echo [1/6] Initializing Visual Studio 2026 build tools...
        call "D:\vs studio\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1
    ) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" (
        echo [1/6] Initializing Visual Studio 2022 build tools...
        call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1
    ) else (
        echo [1/6] MSVC environment already loaded or standard PATH.
    )
) else (
    echo [1/6] Visual Studio build environment already active.
)

REM 2. Compile C++ Native GPU OpenCL/DXGI Static Library
echo [2/6] Compiling C++ Native GPU allocator and DXGI telemetry backend...
pushd cpp
if not exist "build" mkdir "build"
cl /nologo /O2 /MD /EHsc /c /Iinclude src\tritier_gpu_native.cpp /Fobuild\tritier_gpu_native.obj
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to compile C++ native GPU backend.
    popd
    exit /b %ERRORLEVEL%
)
lib /nologo /OUT:build\tritier_sycl.lib build\tritier_gpu_native.obj dxgi.lib
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to link tritier_sycl.lib.
    popd
    exit /b %ERRORLEVEL%
)
popd
echo       C++ Native backend successfully built: cpp\build\tritier_sycl.lib

REM 3. Build Rust Core & FFI Shared Library (cdylib + rlib + CLI binary)
echo [3/6] Building Rust TriTier Core AND CLI with native GPU feature...
cargo build --release --features sycl
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Rust release build failed.
    exit /b %ERRORLEVEL%
)
echo       Rust binaries generated in target\release\

REM 4. Execute Full Acceptance Test Suite (29 unit & acceptance tests)
echo [4/6] Running Rust Unit and Acceptance Tests...
cargo test --features sycl
if %ERRORLEVEL% neq 0 (
    echo [ERROR] TriTier acceptance test suite failed.
    exit /b %ERRORLEVEL%
)
echo       All Rust tests passed successfully!

REM 5. Run Synthetic 10-Billion-Parameter Benchmark
echo [5/6] Running Synthetic 10-Billion Parameter Benchmark (<500MB VRAM verification)...
target\release\tritier.exe benchmark run-10b --operations 64 --config configs\benchmark_10b.json --require-native-gpu
if %ERRORLEVEL% neq 0 (
    echo [ERROR] 10B Parameter benchmark failed.
    exit /b %ERRORLEVEL%
)

REM 6. Run Julia Adapter Verification (if Julia installed)
echo [6/6] Verifying Julia Integration Adapter...
set JULIA_BIN=C:\Users\Woody\AppData\Local\Programs\Julia-1.13.1\bin\julia.exe
if exist "%JULIA_BIN%" (
    "%JULIA_BIN%" julia\test_adapter.jl
) else (
    where julia >nul 2>&1
    if %ERRORLEVEL% equ 0 (
        julia julia\test_adapter.jl
    ) else (
        echo [INFO] Julia binary not found in standard paths; skipping Julia standalone test.
    )
)

echo ===============================================================================
echo TRITIER BUILD AND ACCEPTANCE VERIFICATION COMPLETED SUCCESSFULLY!
echo ===============================================================================
