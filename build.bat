@echo off
REM Navigate to the 2dTerrain project directory
cd 2dTerrain

REM Build the project
dotnet build
IF ERRORLEVEL 1 (
    echo Build failed. Exiting.
    exit /b 1
)

REM Define the output directory
SET OutputDir=bin\Debug\net8.0-windows

REM Create the images folder in the output directory
mkdir "%OutputDir%\images"

REM Copy all files from Images to the new images folder in the output directory without asking for confirmation
xcopy /E /I /Y "Images\*" "%OutputDir%\images\"

REM Run the project
dotnet run --project 2dTerrain.csproj
