@echo off
REM Set up the Visual Studio environment
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

REM Navigate to the project directory
cd "C:\Users\ccw10\source\repos\CudaRuntime1\CudaRuntime1"

REM Compile the CUDA program
nvcc -o vectorexample.exe kernel.cu

REM Run the compiled program
vectorexample.exe

REM Pause to see the output (optional)
pause
