#!/usr/bin/env bash
# Cross-platform setup for Mac/Linux.
# Windows users: see setup.ps1 instead.
set -e

echo "=== Bitext Customer Service Agent — Setup ==="

# 1. Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "[1/4] Creating virtual environment..."
    python3 -m venv .venv
else
    echo "[1/4] Virtual environment already exists — skipping."
fi

# 2. Activate and install
echo "[2/4] Installing dependencies..."
source .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

# 3. Create .env from template if not present
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "[3/4] Created .env from .env.example — fill in your API keys before running."
else
    echo "[3/4] .env already exists — skipping."
fi

# 4. Download dataset
echo "[4/4] Downloading dataset (skips if already present)..."
python download_data.py

echo ""
echo "=== Setup complete ==="
echo "Activate the environment: source .venv/bin/activate"
echo "Run the CLI:              python main.py"
echo "Run LangGraph Studio:     langgraph dev"
