@echo off
REM PDF Translator - All-in-One Script for Windows
REM Auto-setup and run

echo ============================================================
echo PDF TRANSLATOR
echo ============================================================
echo.

REM Check if venv exists
if exist "venv\" (
    echo [INFO] Virtual environment found.
) else (
    echo [SETUP] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment!
        echo [ERROR] Make sure Python is installed and in PATH
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created!
)

REM Activate venv
call venv\Scripts\activate.bat

REM Check if requirements.txt exists
if not exist "requirements.txt" (
    echo [SETUP] Creating requirements.txt...
    (
        echo PyMuPDF==1.23.8
        echo pdfplumber==0.10.3
        echo openai==1.12.0
        echo tqdm==4.66.1
    ) > requirements.txt
    echo [OK] requirements.txt created!
)

REM Check if packages are installed by trying to import them
python -c "import fitz, pdfplumber, openai, tqdm" 2>nul
if errorlevel 1 (
    echo [SETUP] Installing dependencies...
    echo.
    python -m pip install --upgrade pip --quiet
    pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo [ERROR] Failed to install dependencies!
        pause
        exit /b 1
    )
    echo.
    echo [OK] Dependencies installed!
)

REM Check if main script exists
if not exist "pdf_translator.py" (
    echo [ERROR] pdf_translator.py not found!
    echo [ERROR] Please make sure the script is in the same folder
    pause
    exit /b 1
)

echo.
echo [RUN] Starting PDF Translator...
echo ============================================================
echo.

REM Run the translator
python pdf_translator.py

echo.
pause
