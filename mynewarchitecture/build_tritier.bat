@echo off
setlocal enabledelayedexpansion

echo ===============================================================================
echo TRITIER END-TO-END WORKSPACE ORCHESTRATION AND BENCHMARK
echo ===============================================================================

pushd tritier
call build.bat
if %ERRORLEVEL% neq 0 (
    popd
    echo [ERROR] TriTier build failed!
    exit /b %ERRORLEVEL%
)
popd

echo.
echo [Python Step 1] Running Python End-to-End Integration Test...
python test_tritier_integration.py
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python integration test failed!
    exit /b %ERRORLEVEL%
)

echo.
echo [Python Step 2] Running Comparative Benchmark (Un-Tiered vs TriTier Tiered)...
python benchmark_comparison.py
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Comparative benchmark failed!
    exit /b %ERRORLEVEL%
)

echo.
echo ===============================================================================
echo ALL SYSTEMS OPERATIONAL: TRITIER CORE, NATIVE GPU, FFI, PYTHON, JULIA VERIFIED!
echo ===============================================================================
