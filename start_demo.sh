#!/usr/bin/env bash
set -e

# ==============================================================================
# Predictive Processing Multimodal Demo - macOS / Linux Launcher
# ==============================================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo ""
echo "========================================================================="
echo "  Starting Predictive Processing Multimodal Demo (macOS / Linux)"
echo "========================================================================="

# Ensure local bin paths are on PATH
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

# 1. Check for `uv` package manager
if ! command -v uv &> /dev/null; then
    echo "  'uv' package manager not found. Installing uv automatically..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
fi

if ! command -v uv &> /dev/null; then
    echo "  Failed to install 'uv'. Please install uv (brew install uv) and try again."
    exit 1
fi

# 2. Check virtual environment & dependencies
if [ ! -d ".venv" ]; then
    echo "  Initializing virtual environment and syncing dependencies via uv..."
    uv sync
fi

# 3. Check model checkpoints
uv run python download_weights.py

# 4. Check & auto-start Ollama Local LLM engine if installed
OLLAMA_STARTED_BY_SCRIPT=0
if command -v ollama &> /dev/null; then
    echo "  Checking Ollama local LLM engine..."
    if curl -s http://127.0.0.1:11434/api/version &> /dev/null; then
        echo "   [OK] Ollama daemon is already running."
    else
        echo "   Starting Ollama server in background (ollama serve)..."
        ollama serve &> /dev/null &
        OLLAMA_PID=$!
        OLLAMA_STARTED_BY_SCRIPT=1
        for i in {1..10}; do
            if curl -s http://127.0.0.1:11434/api/version &> /dev/null; then
                echo "   [OK] Ollama daemon is ready."
                break
            fi
            sleep 0.5
        done
    fi

    # Discover any installed model in supervisor's Ollama
    OLLAMA_MODEL=$(ollama list 2>/dev/null | awk 'NR>1 {print $1; exit}')
    if [ -n "$OLLAMA_MODEL" ]; then
        echo "   [Model Agnostic] Auto-detected installed Ollama model: '$OLLAMA_MODEL'"
        export OPENAI_MODEL_NAME="$OLLAMA_MODEL"
        echo "   Pre-warming model '$OLLAMA_MODEL' (think disabled)..."
        echo "" | ollama run "$OLLAMA_MODEL" --think=false &> /dev/null || true
    else
        echo "   [Notice] No local models found in Ollama yet."
        echo "   You can install any model of your choice, e.g.: ollama pull qwen2.5:3b"
    fi
fi

cleanup() {
    echo ""
    echo "Shutting down demo processes..."
    if [ "$OLLAMA_STARTED_BY_SCRIPT" -eq 1 ] && [ -n "$OLLAMA_PID" ]; then
        kill "$OLLAMA_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# 5. Open browser once server is ready
(sleep 2.0 && (open "http://127.0.0.1:8000" 2>/dev/null || xdg-open "http://127.0.0.1:8000" 2>/dev/null || true)) &

# 6. Start Unified Demo Server (FastAPI + pyCub + static React Web UI)
echo ""
echo "  Starting Unified Demo Server on http://127.0.0.1:8000..."
echo "  Web UI: http://127.0.0.1:8000"
echo "========================================================================="
uv run uvicorn icub_server:app --host 127.0.0.1 --port 8000
