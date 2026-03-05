"""Transcription module - uses whisper-cli (whisper.cpp) for high-accuracy multilingual transcription."""

import json
import re
import subprocess
import tempfile
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rich.console import Console

from notetaker.config import Settings

console = Console()


# Minimum audio duration (seconds) to bother transcribing.
# Shorter chunks produce more hallucinations.
MIN_TRANSCRIBE_DURATION = 3.0

# Minimum RMS energy to consider audio worth transcribing.
# Below this, it's almost certainly silence/noise from a monitor source.
MIN_TRANSCRIBE_RMS = 0.0005


@dataclass
class TranscriptionSegment:
    """A single transcription segment with timing and metadata."""

    text: str
    start: float
    end: float
    language: str
    confidence: float

    def __str__(self):
        return f"[{self.start:.1f}s-{self.end:.1f}s] ({self.language}) {self.text}"


@dataclass
class TranscriptionResult:
    """Full transcription result for an audio chunk."""

    segments: list[TranscriptionSegment] = field(default_factory=list)
    language: str = ""
    duration: float = 0.0
    processing_time: float = 0.0

    @property
    def full_text(self) -> str:
        return " ".join(seg.text.strip() for seg in self.segments)

    def __str__(self):
        return self.full_text


def _is_hallucination(text: str) -> bool:
    """Detect common Whisper hallucination patterns.

    Whisper hallucinates by:
    - Repeating the same phrase multiple times
    - Generating stock phrases like "Thank you for watching"
    - Producing text that's suspiciously long relative to audio
    """
    text = text.strip()
    if not text:
        return False

    # Split into sentences/phrases
    # Use punctuation or repeated patterns
    phrases = re.split(r'[.!?]+', text)
    phrases = [p.strip() for p in phrases if p.strip()]

    if len(phrases) >= 2:
        # Check if all phrases are identical or near-identical
        unique = set(p.lower() for p in phrases)
        if len(unique) == 1:
            console.print(f"[yellow]  Hallucination detected (repeated phrase): {phrases[0][:60]}...[/yellow]")
            return True

        # Check if most phrases are the same (>= 60% identical)
        from collections import Counter
        counts = Counter(p.lower() for p in phrases)
        most_common_count = counts.most_common(1)[0][1]
        if most_common_count >= len(phrases) * 0.6 and most_common_count >= 2:
            console.print(f"[yellow]  Hallucination detected (phrase repeated {most_common_count}x)[/yellow]")
            return True

    # Common Whisper hallucination phrases (appears when there's no real speech)
    hallucination_patterns = [
        r"(?i)thank you (for watching|for listening)",
        r"(?i)please subscribe",
        r"(?i)see you (next time|in the next)",
        r"(?i)gracias por (ver|escuchar)",
        r"(?i)suscr[ií]be",
        r"(?i)subtitulos? (realizado|por)",
        r"(?i)amara\.org",
    ]
    for pattern in hallucination_patterns:
        if re.search(pattern, text):
            console.print(f"[yellow]  Hallucination detected (stock phrase)[/yellow]")
            return True

    return False


class Transcriber:
    """Whisper-based transcriber using whisper-cli subprocess."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def load_model(self):
        """Verify that whisper-cli and the model file exist."""
        cli_path = Path(self.settings.whisper_cli_path)
        model_path = Path(self.settings.whisper_model_path)

        if not cli_path.exists():
            raise RuntimeError(
                f"whisper-cli not found at {cli_path}\n"
                "Build whisper.cpp or set SUMMAI_WHISPER_CLI_PATH in .env"
            )
        if not model_path.exists():
            raise RuntimeError(
                f"Whisper model not found at {model_path}\n"
                "Download it with: cd /path/to/whisper.cpp && bash models/download-ggml-model.sh large-v3"
            )

        console.print(
            f"[bold cyan]Whisper ready:[/bold cyan] {cli_path.name} + {model_path.name}"
        )

    def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        """Transcribe an audio chunk by calling whisper-cli.

        Args:
            audio: numpy array of float32 audio samples at the configured sample rate.

        Returns:
            TranscriptionResult with segments, language info, and timing.
        """
        duration = len(audio) / self.settings.audio_sample_rate
        rms = float(np.sqrt(np.mean(audio**2)))

        # Skip chunks that are too short (high hallucination risk)
        if duration < MIN_TRANSCRIBE_DURATION:
            console.print(
                f"[dim]Skipping {duration:.1f}s chunk (too short, min={MIN_TRANSCRIBE_DURATION}s)[/dim]"
            )
            return TranscriptionResult(duration=duration)

        # Skip chunks with extremely low energy (silence/noise)
        if rms < MIN_TRANSCRIBE_RMS:
            console.print(
                f"[dim]Skipping {duration:.1f}s chunk (RMS={rms:.6f}, below {MIN_TRANSCRIBE_RMS})[/dim]"
            )
            return TranscriptionResult(duration=duration)

        start = time.time()

        # Write audio to a temporary WAV file (whisper-cli needs a file)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            audio_int16 = (audio * 32767).astype(np.int16)
            with wave.open(tmp_path, "wb") as wf:
                wf.setnchannels(self.settings.audio_channels)
                wf.setsampwidth(2)  # 16-bit = 2 bytes
                wf.setframerate(self.settings.audio_sample_rate)
                wf.writeframes(audio_int16.tobytes())

        try:
            result = self._run_whisper(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
            # Also clean up the JSON output file
            json_path = Path(tmp_path).with_suffix(".json")
            json_path.unlink(missing_ok=True)

        result.duration = duration
        result.processing_time = time.time() - start

        # Post-transcription hallucination check -- mark but keep the text
        if result.full_text.strip() and _is_hallucination(result.full_text):
            for seg in result.segments:
                seg.text = f"[possible hallucination] {seg.text}"

        if result.segments:
            console.print(
                f"[dim]Transcribed {result.duration:.1f}s audio in "
                f"{result.processing_time:.1f}s (lang={result.language}, RMS={rms:.4f})[/dim]"
            )

        return result

    def _run_whisper(self, wav_path: str) -> TranscriptionResult:
        """Run whisper-cli and parse the JSON output."""
        # Build the output file base path (whisper-cli appends .json)
        output_base = str(Path(wav_path).with_suffix(""))

        cmd = [
            self.settings.whisper_cli_path,
            "--model", self.settings.whisper_model_path,
            "--file", wav_path,
            "--output-json",
            "--output-file", output_base,
            "--no-prints",
            "--threads", str(self.settings.whisper_n_threads),
            "--beam-size", str(self.settings.whisper_beam_size),
            "--language", self.settings.whisper_language or "auto",
            "--flash-attn",
            "--suppress-nst",
        ]

        # Add initial prompt for domain vocabulary guidance
        if self.settings.whisper_initial_prompt:
            cmd.extend(["--prompt", self.settings.whisper_initial_prompt])

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )

        if proc.returncode != 0:
            console.print(f"[red]whisper-cli error:[/red] {proc.stderr.strip()}")
            return TranscriptionResult()

        # Parse the JSON output
        json_path = output_base + ".json"
        if not Path(json_path).exists():
            console.print("[red]whisper-cli did not produce JSON output[/red]")
            return TranscriptionResult()

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        language = data.get("result", {}).get("language", "")

        result = TranscriptionResult(language=language)

        for seg in data.get("transcription", []):
            offsets = seg.get("offsets", {})
            start_ms = offsets.get("from", 0)
            end_ms = offsets.get("to", 0)

            result.segments.append(
                TranscriptionSegment(
                    text=seg.get("text", ""),
                    start=start_ms / 1000.0,
                    end=end_ms / 1000.0,
                    language=language,
                    confidence=0.0,  # whisper-cli basic JSON doesn't include per-segment confidence
                )
            )

        return result
