@echo off


if "%~1"=="" (
    echo Usage: %0 [source_file]
    exit /b 1
)


set SOURCE_FILE=%~1


if not exist "%SOURCE_FILE%" (
    echo Source file "%SOURCE_FILE%" not found.
    exit /b 1
)

rem Extract the file name (without extension) from the source file path
for %%F in ("%SOURCE_FILE%") do set FILE_NAME=%%~nF

g++ -I"C:\Program Files (x86)\Microsoft SDKs\MPI\Include" -L"C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64" -o %FILE_NAME%.exe %SOURCE_FILE% -lmsmpi


rem Check if compilation was successful
if %errorlevel% neq 0 (
    echo Compilation failed.
) else (
    echo Compilation successful.
    rem Run the compiled program
    %FILE_NAME%.exe
)
