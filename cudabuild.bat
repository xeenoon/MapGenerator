@echo off
REM Set up the Visual Studio environment
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

REM Navigate to the project directory
cd "C:\Users\ccw10\source\repos\MapGenerator\CudaRuntime1"

REM Compile as a DLL
nvcc --shared -o vectorexample.dll kernel.cu
nvcc --shared -o noise2d.dll noise2d.cu
nvcc --shared -o polygon.dll polygon.cu

REM Copy DLL to the Terrain generator folder
copy /Y vectorexample.dll "..\2dTerrain\bin\Debug\net8.0-windows\"
copy /Y noise2d.dll "..\2dTerrain\bin\Debug\net8.0-windows\"
copy /Y polygon.dll "..\2dTerrain\bin\Debug\net8.0-windows\"