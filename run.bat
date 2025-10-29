#!/bin/bash
# PDF Translator - All-in-One Script for Linux/Mac
# Auto-setup and run

echo "============================================================"
echo "PDF TRANSLATOR"
echo "============================================================"
echo ""

# Check if venv exists
if [ -d "venv" ]; then
    echo "[INFO] Virtual environment found."
else
    echo "[SETUP] Creating virtual environment..."
    python3 -m venv venv
    
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to create virtual environment!"
        echo "[ERROR] Make sure Python 3 is installed"
        exit 1
    fi
    
    echo "[OK] Virtual environment created!"
fi

# Activate venv
source venv/bin/activate

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo "[SETUP] Creating requirements.txt..."
    cat > requirements.txt << 'EOF'
PyMuPDF==1.23.8
pdfplumber==0.10.3
openai==1.12.0
httpx==0.27.0
tqdm==4.66.1
EOF
    echo "[OK] requirements.txt created!"
fi

# Check if packages are installed by trying to import them
python -c "import fitz, pdfplumber, openai, tqdm" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "[SETUP] Installing dependencies..."
    echo ""
    pip install --upgrade pip --quiet
    pip install -r requirements.txt
    
    if [ $? -ne 0 ]; then
        echo ""
        echo "[ERROR] Failed to install dependencies!"
        exit 1
    fi
    
    echo ""
    echo "[OK] Dependencies installed!"
fi

# Check if main script exists
if [ ! -f "pdf_translator.py" ]; then
    echo "[ERROR] pdf_translator.py not found!"
    echo "[ERROR] Please make sure the script is in the same folder"
    exit 1
fi

echo ""
echo "[RUN] Starting PDF Translator..."
echo "============================================================"
echo ""

# Run the translator
python pdf_translator.py
