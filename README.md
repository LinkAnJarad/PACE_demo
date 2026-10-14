
## Quick Start Guide

### macOS (Apple Silicon / Intel)

1. **Install Ollama** (local LLM responses):
   Download and install [Ollama for Mac](https://ollama.com/download/mac).
   Pull a local model:
   ```bash
   ollama pull qwen3.5:2b
   ```

Download Wav2Vec2 weights from https://drive.google.com/file/d/1YiAUoYLF9F8wZye-o1TPu6AcRAo_Z0t7/view?usp=sharing
into `/checkpoints/`

2. **Run the Demo**:
   Double-click `run_demo.command` in Finder!
   *(Or run `./start_demo.sh` from Terminal)*.

The script will automatically:
- Install `uv` if needed.
- Sync Python dependencies (`uv sync`).
- Verify model checkpoints.
- Check and start the Ollama daemon.
- Open your browser at `http://127.0.0.1:8000`.

---

### Windows

1. Ensure [Ollama](https://ollama.com/) or a local OpenAI-compatible server (e.g. `llama-server`) is running.
2. Double-click `start_demo.bat`.

---


## Requirements

- Python 3.11+
- Package manager: [`uv`](https://github.com/astral-sh/uv) (auto-installed by launcher)
- Memory: 8 GB+ RAM (Apple Silicon M1/M2/M3/M4 or NVIDIA GPU supported)
