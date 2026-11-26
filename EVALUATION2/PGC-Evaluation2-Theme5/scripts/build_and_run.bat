@echo off
setlocal
cd /d "%~dp0\.."
echo [1/3] Checking CUDA compiler...
where nvcc || exit /b 1
echo [2/3] Compiling for RTX 5060 Ti (compute capability 12.0)...
nvcc -O2 -arch=sm_120 --ptxas-options=-v src\occupancy_tuning.cu -o occupancy_tuning.exe
if errorlevel 1 exit /b 1
echo [3/3] Running benchmark...
occupancy_tuning.exe
if errorlevel 1 exit /b 1
echo Done. Results are in results\
endlocal
