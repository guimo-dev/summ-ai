<p align="center">
  <img src="logo.svg" alt="Summ-AI" width="480" />
</p>

<p align="center">
  <strong>Fully local AI-powered meeting note-taker.</strong><br>
  Captures system audio from any call app, transcribes in real-time, and generates structured Markdown meeting notes.<br>
  Everything runs on your machine -- no cloud APIs, no data leaves your computer.
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> &middot;
  <a href="#step-by-step-setup">Full Setup Guide</a> &middot;
  <a href="#configuration">Configuration</a> &middot;
  <a href="#how-it-works">How It Works</a>
</p>

---

## What is Summ-AI?

Summ-AI is a vibecoded app that records everything you hear through your speakers or headset during a meeting -- regardless of which app you use (Teams, Zoom, Discord, WhatsApp, Google Meet, browser, etc.) -- and produces structured Markdown notes with topics, decisions, action items, and more.

**Stack:** [whisper.cpp](https://github.com/ggerganov/whisper.cpp) (transcription) + [llama.cpp](https://github.com/ggerganov/llama.cpp) with Qwen3.5-9B (summarization) + ffmpeg + PulseAudio/PipeWire

**Current platform:** Linux with PulseAudio or PipeWire. macOS and Windows support is planned for a future version.

### Features

- Works with **any application** -- no plugins or integrations needed
- **Real-time transcription** using Whisper large-v3 with GPU acceleration
- **Speaker identification** -- LLM-based diarization labels who said what (no extra models needed)
- **Automatic language detection** per chunk (works with any language or mixed-language meetings)
- **Structured notes** in Markdown: topics, decisions, action items, technical references, open questions
- **Hallucination detection** -- marks suspicious Whisper output so you can review it
- **Configurable LLM system prompt** -- customize how notes are generated
- **100% local** -- no cloud, no API keys, no subscriptions


## Requirements

| Requirement | Why |
|---|---|
| **Linux** with PulseAudio or PipeWire | Audio capture uses monitor sources |
| **Python 3.11+** | Runtime |
| **ffmpeg** | Audio capture from monitor sources |
| **git** | To clone repos |
| **CMake + C++ compiler** | To build whisper.cpp and llama.cpp |

### Resource usage

Both models run on GPU simultaneously, for testing purposes I have used successfully:

| Component | VRAM | Notes |
|---|---|---|
| Whisper large-v3 | ~3 GB | Transcribes 30s of audio in ~1-3s |
| Qwen3.5-9B Q4_K_M | ~6 GB | Generates notes in ~3-5s |
| **Total** | **~9 GB** | Fits on a 12-16 GB GPU |


## Quick start

If you already have whisper.cpp and llama.cpp built, and a Whisper model downloaded:

```bash
git clone https://github.com/guimo-dev/summ-ai.git
cd summ-ai

python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# Edit .env with your paths (see Configuration)
cp .env.example .env
nano .env   # set SUMMAI_WHISPER_CLI_PATH, SUMMAI_WHISPER_MODEL_PATH, SUMMAI_LLAMA_SERVER_PATH

# Run
summ-ai
```

Press **Ctrl+C** when the meeting ends -- it generates the notes and saves everything.


## Step-by-step setup

This is the full setup from zero. Follow these steps in order.

### Step 1: Install system dependencies

You need ffmpeg, a C++ compiler, CMake, and the CUDA toolkit.

```bash
# Arch Linux
sudo pacman -S base-devel cmake ffmpeg cuda

# Ubuntu / Debian
sudo apt install build-essential cmake ffmpeg nvidia-cuda-toolkit
```

### Step 2: Build whisper.cpp

whisper.cpp provides `whisper-cli`, the tool that transcribes audio locally on your GPU.

```bash
# Pick a directory to keep your AI tools (example: ~/ai-tools)
mkdir -p ~/ai-tools && cd ~/ai-tools

git clone https://github.com/ggerganov/whisper.cpp.git
cd whisper.cpp

# Build with CUDA (GPU acceleration)
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)

# Download the Whisper large-v3 model (~3 GB)
bash models/download-ggml-model.sh large-v3
```

Verify the build:

```bash
# These two files must exist:
ls build/bin/whisper-cli       # the binary
ls models/ggml-large-v3.bin    # the model
```

Take note of the full paths -- you will need them for the `.env` file. For example:
```
~/ai-tools/whisper.cpp/build/bin/whisper-cli
~/ai-tools/whisper.cpp/models/ggml-large-v3.bin
```

### Step 3: Build llama.cpp

llama.cpp provides `llama-server`, which runs the Qwen3.5-9B language model as a local HTTP server for summarization.

```bash
cd ~/ai-tools

git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# Build with CUDA
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
```

Verify the build:

```bash
ls build/bin/llama-server
```

Take note of the full path. For example:
```
~/ai-tools/llama.cpp/build/bin/llama-server
```

> **Note:** You do NOT need to download the LLM model manually. On first run, `llama-server` will auto-download the Qwen3.5-9B model (~5.3 GB) from HuggingFace. Subsequent starts load it from cache in seconds.

### Step 4: Clone Summ-AI and create the Python environment

```bash
git clone https://github.com/YOUR_USERNAME/summ-ai.git
cd summ-ai

# Create a virtual environment (only needed once)
python3 -m venv .venv

# Activate it (you need this in every new terminal)
source .venv/bin/activate

# Install Summ-AI and its dependencies
pip install -e .
```

> **What does `source .venv/bin/activate` do?**
> It tells your terminal to use the Python inside `.venv/` instead of the system Python. Packages installed with `pip` go into `.venv/` and don't affect anything else on your system. Type `deactivate` to go back to normal.

### Step 5: Configure your paths

Copy the example environment file and edit it with the paths from Steps 2 and 3:

```bash
cp .env.example .env
```

Open `.env` in your editor and set **these three paths** to match where you built whisper.cpp and llama.cpp:

```bash
SUMMAI_WHISPER_CLI_PATH=/home/youruser/ai-tools/whisper.cpp/build/bin/whisper-cli
SUMMAI_WHISPER_MODEL_PATH=/home/youruser/ai-tools/whisper.cpp/models/ggml-large-v3.bin
SUMMAI_LLAMA_SERVER_PATH=/home/youruser/ai-tools/llama.cpp/build/bin/llama-server
```

Use **absolute paths** (starting with `/`). The `~` shortcut works too.

Everything else in `.env` has sensible defaults -- you only need to change these three paths.

### Step 6: Run Summ-AI

```bash
cd /path/to/summ-ai
source .venv/bin/activate
summ-ai
```

The first time you run it, `llama-server` will download the Qwen3.5-9B model (~5.3 GB). This only happens once -- subsequent starts take about 4-5 seconds.

Join your meeting. Summ-AI captures everything you hear through your headset or speakers. Press **Ctrl+C** when the meeting ends -- it saves a transcript and generates structured notes.


## Usage

```
summ-ai [OPTIONS]
```

| Option | Description |
|---|---|
| (no options) | Start with auto-detected audio source |
| `--list-sources` / `-l` | List available audio monitor sources |
| `--source NAME` / `-s NAME` | Use a specific PulseAudio source |
| `--language CODE` | Force transcription language (`en`, `es`, etc.). Default: auto-detect |
| `--save-audio` | Also save the raw recording as a WAV file |
| `--output-dir DIR` / `-o DIR` | Set output directory (default: `./meeting_notes`) |
| `--chunk-duration SEC` | Audio chunk size in seconds (default: 30) |
| `--no-diarize` | Disable speaker identification (faster processing) |
| `--llm-model REPO` | Use a different HuggingFace model for summarization |

### Examples

```bash
# List what audio sources are available
summ-ai --list-sources

# Force Spanish transcription
summ-ai --language es

# Save both notes and the raw audio recording
summ-ai --save-audio

# Use a specific audio source
summ-ai --source alsa_output.usb-headset.monitor

# Save to a custom directory
summ-ai -o ~/Documents/meetings
```


## Output

Each session creates a timestamped folder:

```
meeting_notes/
  meeting_20260305_143000/
    transcript.md     # Raw transcript with timestamps and language tags
    notes.md          # Structured meeting notes
    recording.wav     # Full audio (only if --save-audio was used)
```

The **notes.md** contains structured sections:
- Key topics discussed
- Detailed discussion summary
- Decisions made
- Action items (with owners when identifiable)
- Technical references (tools, systems, APIs, etc.)
- Open questions


## How it works

### Audio capture

Summ-AI records from a **PulseAudio/PipeWire monitor source** -- a virtual microphone that captures everything playing through an audio output device. This means:

- It works with **any application** (Teams, Zoom, Discord, browser, etc.)
- No app-specific plugins or integrations needed
- It captures exactly what you hear through your headset/speakers

By default, it auto-detects the **active** output device (the one currently playing audio). If nothing is playing yet, it falls back to the default output device's monitor.

> **Why ffmpeg?** On PipeWire systems, most audio tools (`parecord`, `pw-record`, `sounddevice`) return silence when reading from monitor sources. ffmpeg handles them correctly.

### Transcription pipeline

Audio is captured in configurable chunks (default: 30 seconds), each sent to `whisper-cli` for GPU-accelerated transcription. Whisper large-v3 auto-detects the language per chunk, so mixed-language meetings work out of the box.

A hallucination detector flags suspicious Whisper output (repeated phrases, stock phrases like "Thank you for watching") with a `[possible hallucination]` prefix. The text is kept so you can review it yourself.

### Summarization

When you press Ctrl+C, the full transcript is sent to Qwen3.5-9B running locally via `llama-server`. For long meetings, intermediate summaries are generated periodically to avoid exceeding the context window.

### Speaker identification

Summ-AI uses the LLM (Qwen3.5) to identify who said what in the transcript -- no extra models or dependencies required. After each chunk is transcribed, it is sent to the LLM with a diarization prompt that:

- Labels each speech turn with a speaker name (when names are mentioned in conversation) or a consistent `Speaker 1`, `Speaker 2`, etc. label
- Maintains a running speaker context across chunks so speaker labels stay consistent throughout the meeting
- Works for any language, including mixed-language meetings

This approach works best when speakers address each other by name or have distinct roles/topics. In ambiguous cases (similar speakers, same topic, no identifying context), the LLM will still attempt to separate turns based on conversational patterns.

To disable speaker identification (for faster processing):
```bash
summ-ai --no-diarize
# or in .env:
SUMMAI_DIARIZE=false
```

### Virtual audio cable (optional, for cleaner capture)

If your call app routes audio to a different device, you can set up a dedicated virtual sink:

```bash
# Create a virtual sink called "MeetingCapture"
pactl load-module module-null-sink sink_name=meeting_capture \
  sink_properties=device.description='MeetingCapture'

# Loop its output back so you can still hear it
pactl load-module module-loopback source=meeting_capture.monitor latency_msec=1

# Set your call app's audio output to "MeetingCapture" in system sound settings
# Then tell Summ-AI to use it:
summ-ai --source meeting_capture.monitor
```


## Configuration

All settings can be set via environment variables (prefix `SUMMAI_`) or in a `.env` file. See `.env.example` for the full list with descriptions.

### Required settings

These three paths must point to your local builds of whisper.cpp and llama.cpp:

| Variable | Description |
|---|---|
| `SUMMAI_WHISPER_CLI_PATH` | Path to the `whisper-cli` binary |
| `SUMMAI_WHISPER_MODEL_PATH` | Path to the Whisper GGML model file |
| `SUMMAI_LLAMA_SERVER_PATH` | Path to the `llama-server` binary |

### Optional settings

| Variable | Default | Description |
|---|---|---|
| `SUMMAI_LLM_HF_REPO` | `unsloth/Qwen3.5-9B-GGUF:Q4_K_M` | HuggingFace model ID for summarization |
| `SUMMAI_LLM_PORT` | `8079` | Port for llama-server |
| `SUMMAI_WHISPER_LANGUAGE` | (auto-detect) | Force language: `en`, `es`, etc. |
| `SUMMAI_WHISPER_INITIAL_PROMPT` | (empty) | Domain vocabulary hint for Whisper |
| `SUMMAI_SYSTEM_PROMPT` | (built-in) | Custom system prompt for the LLM summarizer |
| `SUMMAI_SILENCE_THRESHOLD` | `0.001` | RMS below this = silence |
| `SUMMAI_DIARIZE` | `true` | Enable LLM-based speaker identification |
| `SUMMAI_AUDIO_CHUNK_DURATION` | `30` | Seconds per transcription chunk |
| `SUMMAI_OUTPUT_DIR` | `./meeting_notes` | Where to save output |
| `SUMMAI_SAVE_AUDIO` | `false` | Save raw audio recording |
| `SUMMAI_SAVE_TRANSCRIPT` | `true` | Save raw transcript file |
| `SUMMAI_SUMMARIZE_EVERY_N_CHUNKS` | `10` | Intermediate summary frequency |


## Troubleshooting

**"whisper-cli not found" / "llama-server not found"**
Set the correct paths in your `.env` file:
```bash
SUMMAI_WHISPER_CLI_PATH=/full/path/to/whisper-cli
SUMMAI_LLAMA_SERVER_PATH=/full/path/to/llama-server
```

**"No monitor source found"**
PulseAudio/PipeWire is not running or has no output devices:
```bash
pactl list sources short | grep monitor
```

**llama-server takes a long time on first start**
The model (~5.3 GB) is being downloaded from HuggingFace. Check progress in the terminal. After the first run, it loads in seconds.

**Whisper hallucinates text during silence**
This is normal Whisper behavior. The silence filter (RMS threshold) skips quiet chunks. If too many silent chunks get through, increase the threshold:
```bash
SUMMAI_SILENCE_THRESHOLD=0.005
```

**Real speech is being filtered as silence**
Monitor sources have low audio levels (RMS ~0.002 for normal speech). Lower the threshold:
```bash
SUMMAI_SILENCE_THRESHOLD=0.0005
```

**"ffmpeg not found"**
```bash
# Arch Linux
sudo pacman -S ffmpeg
# Ubuntu / Debian
sudo apt install ffmpeg
```

**No speech detected (but audio was playing)**
Your app may route audio to a different output device. List sources and pick the right one:
```bash
summ-ai --list-sources
summ-ai --source <your.monitor.name>
```

**Python command not found after install**
Activate the virtual environment first:
```bash
source /path/to/summ-ai/.venv/bin/activate
```

You can add an alias to your shell config for convenience:
```bash
# Add to ~/.bashrc or ~/.zshrc:
alias summ-ai="source /path/to/summ-ai/.venv/bin/activate && summ-ai"
```


## Platform support

| Platform | Status |
|---|---|
| Linux (PulseAudio/PipeWire) | Supported |
| macOS (CoreAudio) | Planned |
| Windows (WASAPI) | Planned |

Contributions for macOS and Windows audio capture are welcome.


## License

MIT
