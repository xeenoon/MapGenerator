@echo off
REM Navigate to the 2dTerrain project directory
cd 2dTerrain

REM Build the project
dotnet build
IF ERRORLEVEL 1 (
    echo Build failed. Exiting.
    exit /b 1
)

REM Run the project
dotnet run --project 2dTerrain.csproj
