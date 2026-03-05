#!/usr/bin/env bash
# =============================================================================
# Summ-AI - Setup Script
# =============================================================================
set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }
step()  { echo -e "\n${CYAN}==> $*${NC}"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Step 1: System dependencies -------------------------------------------
step "Checking system dependencies..."

MISSING=()
for cmd in python3 pip3 pactl ffmpeg; do
    if ! command -v "$cmd" &>/dev/null; then
        MISSING+=("$cmd")
    fi
done

if [ ${#MISSING[@]} -gt 0 ]; then
    error "Missing commands: ${MISSING[*]}"
    echo "Install with:"
    echo "  Arch:   sudo pacman -S python ffmpeg pulseaudio-utils"
    echo "  Ubuntu: sudo apt install python3 python3-pip ffmpeg pulseaudio-utils"
    exit 1
fi

info "System dependencies OK."

# ---- Step 2: Check for whisper-cli and llama-server -------------------------
step "Checking for whisper-cli and llama-server..."

# Try to find binaries from .env or environment
ENV_FILE="$SCRIPT_DIR/.env"
WHISPER_CLI="${SUMMAI_WHISPER_CLI_PATH:-}"
LLAMA_SERVER="${SUMMAI_LLAMA_SERVER_PATH:-}"
WHISPER_MODEL="${SUMMAI_WHISPER_MODEL_PATH:-}"

# Source .env if it exists and vars are not already set
if [ -f "$ENV_FILE" ]; then
    while IFS='=' read -r key value; do
        # Skip comments and empty lines
        [[ "$key" =~ ^[[:space:]]*# ]] && continue
        [[ -z "$key" ]] && continue
        key="$(echo "$key" | xargs)"
        value="$(echo "$value" | xargs)"
        case "$key" in
            SUMMAI_WHISPER_CLI_PATH)  [ -z "$WHISPER_CLI" ] && WHISPER_CLI="$value" ;;
            SUMMAI_LLAMA_SERVER_PATH) [ -z "$LLAMA_SERVER" ] && LLAMA_SERVER="$value" ;;
            SUMMAI_WHISPER_MODEL_PATH) [ -z "$WHISPER_MODEL" ] && WHISPER_MODEL="$value" ;;
        esac
    done < "$ENV_FILE"
fi

# Expand ~ in paths
WHISPER_CLI="${WHISPER_CLI/#\~/$HOME}"
LLAMA_SERVER="${LLAMA_SERVER/#\~/$HOME}"
WHISPER_MODEL="${WHISPER_MODEL/#\~/$HOME}"

if [ -n "$WHISPER_CLI" ] && [ -x "$WHISPER_CLI" ]; then
    info "whisper-cli found: $WHISPER_CLI"
else
    if [ -n "$WHISPER_CLI" ]; then
        warn "whisper-cli not found at: $WHISPER_CLI"
    else
        warn "SUMMAI_WHISPER_CLI_PATH not set"
    fi
    echo "  Build whisper.cpp and set SUMMAI_WHISPER_CLI_PATH in .env"
    echo "  See README.md Step 2 for instructions."
fi

if [ -n "$LLAMA_SERVER" ] && [ -x "$LLAMA_SERVER" ]; then
    info "llama-server found: $LLAMA_SERVER"
else
    if [ -n "$LLAMA_SERVER" ]; then
        warn "llama-server not found at: $LLAMA_SERVER"
    else
        warn "SUMMAI_LLAMA_SERVER_PATH not set"
    fi
    echo "  Build llama.cpp and set SUMMAI_LLAMA_SERVER_PATH in .env"
    echo "  See README.md Step 3 for instructions."
fi

if [ -n "$WHISPER_MODEL" ] && [ -f "$WHISPER_MODEL" ]; then
    info "Whisper model found: $WHISPER_MODEL"
else
    if [ -n "$WHISPER_MODEL" ]; then
        warn "Whisper model not found at: $WHISPER_MODEL"
    else
        warn "SUMMAI_WHISPER_MODEL_PATH not set"
    fi
    echo "  Download with: cd /path/to/whisper.cpp && bash models/download-ggml-model.sh large-v3"
fi

# ---- Step 3: Check GPU support ----------------------------------------------
step "Checking GPU support..."

if command -v nvidia-smi &>/dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "")
    if [ -n "$GPU_INFO" ]; then
        info "GPU detected: $GPU_INFO"
    else
        warn "nvidia-smi found but no GPU info available"
    fi
else
    warn "nvidia-smi not found. GPU acceleration may not be available."
fi

# ---- Step 4: Python virtual environment -------------------------------------
step "Setting up Python virtual environment..."

VENV_DIR="$SCRIPT_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    info "Created virtual environment at $VENV_DIR"
else
    info "Virtual environment already exists."
fi

source "$VENV_DIR/bin/activate"
pip install --upgrade pip -q

# ---- Step 5: Install Python dependencies ------------------------------------
step "Installing Python dependencies..."
pip install -e "$SCRIPT_DIR" 2>&1 | tail -5
info "Python dependencies installed."

# ---- Step 6: LLM model info ------------------------------------------------
step "LLM model info..."

echo ""
echo "  The Qwen3.5-9B model will be auto-downloaded on first run via llama-server."
echo "  Default model: unsloth/Qwen3.5-9B-GGUF:Q4_K_M (~5.3 GB)"
echo ""

# ---- Step 7: Audio setup guidance -------------------------------------------
step "Audio setup for meeting capture..."

echo ""
echo "  Summ-AI captures audio from a PulseAudio/PipeWire monitor source."
echo ""
echo "  Option A: Use existing monitor (default -- works out of the box):"
echo "    Run:  summ-ai --list-sources"
echo "    Look for sources marked [DEFAULT] or [ACTIVE]"
echo ""
echo "  Option B: Create a virtual audio cable (recommended for cleanest capture):"
echo "    pactl load-module module-null-sink sink_name=meeting_capture sink_properties=device.description='MeetingCapture'"
echo "    pactl load-module module-loopback source=meeting_capture.monitor latency_msec=1"
echo "    Then set your call app's audio output to 'MeetingCapture' in system sound settings."
echo ""

# ---- Done -------------------------------------------------------------------
step "Setup complete!"
echo ""
echo "  Quick start:"
echo "    source $VENV_DIR/bin/activate"
echo "    summ-ai --list-sources    # Find available monitor sources"
echo "    summ-ai                   # Start recording (auto-detects default)"
echo "    summ-ai --source NAME     # Use a specific PulseAudio source"
echo ""

if [ ! -f "$SCRIPT_DIR/.env" ]; then
    echo "  IMPORTANT: Create your .env file with the required paths:"
    echo "    cp $SCRIPT_DIR/.env.example $SCRIPT_DIR/.env"
    echo "    # Edit .env and set SUMMAI_WHISPER_CLI_PATH, SUMMAI_WHISPER_MODEL_PATH, SUMMAI_LLAMA_SERVER_PATH"
    echo ""
fi
