# Cross-platform setup for Windows (PowerShell).
# Mac/Linux users: run ./setup.sh instead.

Write-Host "=== Bitext Customer Service Agent - Setup ===" -ForegroundColor Cyan

# 1. Create virtual environment
if (-Not (Test-Path ".venv")) {
    Write-Host "[1/4] Creating virtual environment..."
    python -m venv .venv
} else {
    Write-Host "[1/4] Virtual environment already exists - skipping."
}

# 2. Activate and install
Write-Host "[2/4] Installing dependencies..."
.\.venv\Scripts\Activate.ps1
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

# 3. Create .env from template
if (-Not (Test-Path ".env")) {
    Copy-Item .env.example .env
    Write-Host "[3/4] Created .env from .env.example - fill in your API keys before running."
} else {
    Write-Host "[3/4] .env already exists - skipping."
}

# 4. Download dataset
Write-Host "[4/4] Downloading dataset (skips if already present)..."
python download_data.py

Write-Host ""
Write-Host "=== Setup complete ===" -ForegroundColor Green
Write-Host "Activate the environment: .\.venv\Scripts\Activate.ps1"
Write-Host "Run the CLI:              python main.py"
Write-Host "Run LangGraph Studio:     langgraph dev"
