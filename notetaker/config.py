"""Configuration management for Summ-AI."""

from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field, model_validator


# Default system prompt for the LLM summarizer.
# Users can override this entirely via SUMMAI_SYSTEM_PROMPT env var or .env file.
DEFAULT_SYSTEM_PROMPT = """\
You are a precise meeting note-taker AI.

Your task is to produce well-structured meeting notes from a transcript. \
The meeting may be conducted in any language or a mix of languages.

CRITICAL RULES:
1. Preserve ALL technical terms, proper nouns, and domain-specific vocabulary exactly as spoken.
2. Write the notes in the SAME LANGUAGE as the majority of the transcript. \
   If the meeting is mixed, use the dominant language but keep technical terms as-is.
3. Never invent information not present in the transcript.
4. If something is unclear in the transcript, mark it as [unclear] rather than guessing.
5. Attribute statements to speakers when identifiable.

OUTPUT FORMAT (Markdown):
```
# Meeting Notes - [Date/Topic if identifiable]

## Key Topics Discussed
- Bullet points of main topics

## Detailed Discussion
Organized summary of the conversation with technical details preserved.

## Decisions Made
- List of decisions reached during the meeting

## Action Items
- [ ] Action item with responsible person if identifiable
- [ ] Another action item

## Technical References
- Specific tools, systems, APIs, or technologies mentioned
- Links, documents, or resources referenced

## Open Questions
- Unresolved questions or items needing follow-up
```
"""


class Settings(BaseSettings):
    """Application settings with environment variable support.

    Required settings (must be set in .env or environment):
        - whisper_cli_path: path to whisper-cli binary
        - whisper_model_path: path to Whisper GGML model
        - llama_server_path: path to llama-server binary
    """

    # Audio capture
    audio_sample_rate: int = Field(default=16000, description="Audio sample rate in Hz")
    audio_channels: int = Field(default=1, description="Mono audio for transcription")
    audio_chunk_duration: int = Field(
        default=30, description="Duration of each audio chunk in seconds for transcription"
    )
    silence_threshold: float = Field(
        default=0.001,
        description="RMS threshold below which audio is considered silence. "
        "Monitor sources have low levels (~0.002 RMS for normal audio), "
        "so this must be well below that.",
    )
    silence_duration: float = Field(
        default=2.0, description="Seconds of silence before splitting chunks"
    )

    # Whisper transcription (whisper-cli from whisper.cpp)
    whisper_cli_path: str = Field(
        default="",
        description="Path to the whisper-cli binary (REQUIRED)",
    )
    whisper_model_path: str = Field(
        default="",
        description="Path to the Whisper GGML model file (REQUIRED)",
    )
    whisper_n_threads: int = Field(
        default=4, description="Number of CPU threads for whisper.cpp"
    )
    whisper_beam_size: int = Field(default=5, description="Beam size for decoding")
    whisper_language: str | None = Field(
        default=None,
        description="Force language (None = auto-detect, 'es' or 'en' to force)",
    )
    whisper_initial_prompt: str = Field(
        default="",
        description="Initial prompt to guide Whisper with domain vocabulary (leave empty for general use)",
    )

    # LLM summarization (llama-server from llama.cpp, OpenAI-compatible API)
    llama_server_path: str = Field(
        default="",
        description="Path to the llama-server binary (REQUIRED)",
    )
    llm_hf_repo: str = Field(
        default="unsloth/Qwen3.5-9B-GGUF:Q4_K_M",
        description="HuggingFace repo:quant for -hf flag (auto-downloads on first run)",
    )
    llm_n_gpu_layers: int = Field(
        default=-1, description="Number of layers to offload to GPU (-1 = all)"
    )
    llm_n_ctx: int = Field(
        default=16384, description="Context window size for the LLM"
    )
    llm_host: str = Field(
        default="127.0.0.1", description="Host for llama-server to bind to"
    )
    llm_port: int = Field(
        default=8079, description="Port for llama-server (use non-standard to avoid conflicts)"
    )
    system_prompt: str = Field(
        default=DEFAULT_SYSTEM_PROMPT,
        description="System prompt for the LLM summarizer. Override to customize note-taking behavior.",
    )
    summarize_every_n_chunks: int = Field(
        default=10,
        description="Produce an intermediate summary every N transcription chunks",
    )
    diarize: bool = Field(
        default=True,
        description="Enable LLM-based speaker diarization (identify who said what). "
        "Uses the already-running llama-server to infer speaker identity from context.",
    )

    # Output
    output_dir: Path = Field(
        default=Path("./meeting_notes"), description="Directory for meeting notes output"
    )
    save_transcript: bool = Field(
        default=True, description="Also save the raw transcript alongside the summary"
    )
    save_audio: bool = Field(
        default=False, description="Save the recorded audio as WAV file"
    )

    model_config = {"env_prefix": "SUMMAI_", "env_file": ".env"}

    @model_validator(mode="after")
    def _check_required_paths(self) -> "Settings":
        """Validate that the three required paths are set."""
        missing = []
        if not self.whisper_cli_path:
            missing.append("SUMMAI_WHISPER_CLI_PATH")
        if not self.whisper_model_path:
            missing.append("SUMMAI_WHISPER_MODEL_PATH")
        if not self.llama_server_path:
            missing.append("SUMMAI_LLAMA_SERVER_PATH")

        if missing:
            raise ValueError(
                f"Required settings not configured: {', '.join(missing)}\n"
                "Set them in your .env file or as environment variables.\n"
                "See .env.example for details, or run: cp .env.example .env"
            )
        return self
