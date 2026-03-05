"""Summ-AI main application - orchestrates audio capture, transcription, and summarization."""

import os
import signal
import subprocess
import sys
import time
import argparse
import atexit
from datetime import datetime
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.panel import Panel

from notetaker.config import Settings
from notetaker.audio import AudioCapture
from notetaker.transcriber import Transcriber
from notetaker.summarizer import Summarizer

console = Console()


class LlamaServerManager:
    """Manages a llama-server subprocess lifecycle."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._process: subprocess.Popen | None = None

    def start(self) -> None:
        """Start llama-server as a background subprocess.

        Starts in its own process group so that Ctrl+C (SIGINT) sent to the
        terminal's foreground group does NOT kill llama-server. We need it
        alive after Ctrl+C to generate the final summary.
        """
        server_path = Path(self.settings.llama_server_path)
        if not server_path.exists():
            raise RuntimeError(
                f"llama-server not found at {server_path}\n"
                "Build llama.cpp or set SUMMAI_LLAMA_SERVER_PATH in .env"
            )

        cmd = [
            str(server_path),
            "-hf", self.settings.llm_hf_repo,
            "--host", self.settings.llm_host,
            "--port", str(self.settings.llm_port),
            "-ngl", str(self.settings.llm_n_gpu_layers),
            "-c", str(self.settings.llm_n_ctx),
            "--no-webui",
        ]

        console.print(
            f"[bold cyan]Starting llama-server...[/bold cyan]\n"
            f"  Model: {self.settings.llm_hf_repo}\n"
            f"  Endpoint: http://{self.settings.llm_host}:{self.settings.llm_port}"
        )

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,  # own process group, immune to Ctrl+C
        )

        # Register cleanup on exit
        atexit.register(self.stop)

    def stop(self) -> None:
        """Stop the llama-server subprocess."""
        if self._process is not None and self._process.poll() is None:
            console.print("[dim]Stopping llama-server...[/dim]")
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
            self._process = None

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None


class MeetingSession:
    """Manages a complete meeting recording and note-taking session."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.audio = AudioCapture(settings)
        self.transcriber = Transcriber(settings)
        self.summarizer = Summarizer(settings)
        self._server_manager = LlamaServerManager(settings)

        self._transcript_segments: list[str] = []
        self._chunk_count = 0
        self._running = False
        self._meeting_start: datetime | None = None
        self._total_audio_duration = 0.0

    def _setup_output_dir(self) -> Path:
        """Create the output directory for this meeting session."""
        self.settings.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = self.settings.output_dir / f"meeting_{timestamp}"
        session_dir.mkdir(exist_ok=True)
        return session_dir

    def _save_transcript(self, session_dir: Path) -> Path:
        """Save the raw transcript to a file."""
        transcript_path = session_dir / "transcript.md"
        assert self._meeting_start is not None
        content = f"# Meeting Transcript\n\n"
        content += f"**Date:** {self._meeting_start.strftime('%Y-%m-%d %H:%M')}\n"
        content += f"**Duration:** {self._total_audio_duration / 60:.1f} minutes\n\n"
        content += "---\n\n"
        content += "\n\n".join(self._transcript_segments)
        transcript_path.write_text(content, encoding="utf-8")
        return transcript_path

    def _save_notes(self, session_dir: Path, notes: str) -> Path:
        """Save the structured meeting notes."""
        notes_path = session_dir / "notes.md"
        notes_path.write_text(notes, encoding="utf-8")
        return notes_path

    def _process_audio_chunk(self, audio: np.ndarray) -> str | None:
        """Process a single audio chunk: transcribe, optionally diarize, and accumulate."""
        result = self.transcriber.transcribe(audio)

        if not result.full_text.strip():
            return None

        self._chunk_count += 1
        self._total_audio_duration += result.duration

        text = result.full_text

        # Speaker diarization: send transcript through LLM to add speaker labels
        if self.settings.diarize:
            diarized = self.summarizer.diarize_transcript(text)
            if diarized != text:
                text = diarized
                # Show identified speakers on first discovery
                ctx = self.summarizer.speaker_context
                if ctx.speakers and self._chunk_count <= 3:
                    speakers = ", ".join(ctx.speakers.keys())
                    console.print(f"  [dim]Speakers: {speakers}[/dim]")

        # Format the segment with timestamp offset
        offset_min = self._total_audio_duration / 60
        segment_text = f"**[{offset_min:.1f} min | {result.language}]** {text}"
        self._transcript_segments.append(segment_text)

        # Print live transcription
        lang_color = "blue" if result.language == "en" else "yellow"
        console.print(
            f"  [{lang_color}][{result.language}][/{lang_color}] {text}"
        )

        # Periodically generate intermediate summaries
        if (
            self._chunk_count > 0
            and self._chunk_count % self.settings.summarize_every_n_chunks == 0
        ):
            recent = "\n".join(
                self._transcript_segments[-self.settings.summarize_every_n_chunks:]
            )
            self.summarizer.intermediate_summary(recent)

        return text

    def run(self, source: str | None = None):
        """Run the meeting note-taking session.

        Args:
            source: PulseAudio source name to capture from, or None for auto-detect.
        """
        self._meeting_start = datetime.now()
        session_dir = self._setup_output_dir()
        self._running = True
        self._ctrl_c_count = 0

        # Signal handler for graceful shutdown
        def signal_handler(sig, frame):
            self._ctrl_c_count += 1
            if self._ctrl_c_count == 1:
                console.print("\n[bold yellow]Stopping recording... (press Ctrl+C again to force quit)[/bold yellow]")
                self._running = False
            else:
                console.print("\n[bold red]Force quitting...[/bold red]")
                self._server_manager.stop()
                sys.exit(1)

        signal.signal(signal.SIGINT, signal_handler)

        # Display startup banner
        diarize_status = "[green]enabled[/green]" if self.settings.diarize else "[dim]disabled[/dim]"
        console.print(
            Panel.fit(
                "[bold green]Summ-AI[/bold green]\n"
                f"Whisper: whisper-cli + {Path(self.settings.whisper_model_path).name}\n"
                f"LLM: llama-server + {self.settings.llm_hf_repo}\n"
                f"Speaker ID: {diarize_status}\n"
                f"Output: {session_dir}\n"
                f"\nPress [bold]Ctrl+C[/bold] to stop recording and generate notes.",
                title="Session Started",
                border_style="green",
            )
        )

        # Verify whisper-cli setup
        self.transcriber.load_model()

        # Start llama-server for summarization
        self._server_manager.start()
        console.print("[dim]Waiting for llama-server to load model...[/dim]")
        if not self.summarizer.wait_for_server(timeout=300):
            if not self._server_manager.is_running:
                console.print(
                    "[bold red]llama-server exited unexpectedly. "
                    "Check that the model can be downloaded.[/bold red]"
                )
            else:
                console.print(
                    "[bold red]llama-server timed out loading model.[/bold red]"
                )
            self._server_manager.stop()
            return

        # Start audio capture from system audio (monitor source)
        try:
            self.audio.start(source=source)
        except RuntimeError as e:
            console.print(f"[bold red]{e}[/bold red]")
            self._server_manager.stop()
            return

        console.print(
            "[bold green]Listening... Audio from any app will be captured.[/bold green]\n"
        )

        # Main processing loop
        try:
            while self._running:
                chunk = self.audio.get_chunk(timeout=1.0)
                if chunk is not None:
                    try:
                        self._process_audio_chunk(chunk)
                    except Exception as e:
                        console.print(f"[bold red]Error processing chunk: {e}[/bold red]")
                        console.print("[dim]Continuing to next chunk...[/dim]")
        finally:
            # Stop audio capture first -- this also flushes remaining audio
            self.audio.stop()

            # Process any remaining chunks that were queued during shutdown
            remaining = 0
            while True:
                chunk = self.audio.get_chunk(timeout=0.1)
                if chunk is None:
                    break
                try:
                    self._process_audio_chunk(chunk)
                    remaining += 1
                except Exception:
                    pass
            if remaining:
                console.print(f"[dim]Processed {remaining} remaining chunk(s)[/dim]")

            # Save audio if configured
            if self.settings.save_audio:
                self.audio.save_full_audio(session_dir / "recording.wav")

            # Generate final output
            if self._transcript_segments:
                console.print("\n[bold cyan]Processing final meeting notes...[/bold cyan]")

                # Save raw transcript
                if self.settings.save_transcript:
                    transcript_path = self._save_transcript(session_dir)
                    console.print(f"[green]Transcript saved:[/green] {transcript_path}")

                # Verify llama-server is still alive before attempting summary
                if self._server_manager.is_running:
                    full_transcript = "\n\n".join(self._transcript_segments)
                    try:
                        notes = self.summarizer.final_summary(full_transcript)
                        notes_path = self._save_notes(session_dir, notes)
                        console.print(f"[bold green]Meeting notes saved:[/bold green] {notes_path}")
                    except Exception as e:
                        console.print(
                            f"[bold red]Failed to generate summary: {e}[/bold red]\n"
                            "[yellow]Transcript was saved successfully.[/yellow]"
                        )
                else:
                    console.print(
                        "[bold red]llama-server is not running -- cannot generate summary.[/bold red]\n"
                        "[yellow]Transcript was saved successfully.[/yellow]"
                    )

                console.print(
                    Panel.fit(
                        f"Duration: {self._total_audio_duration / 60:.1f} minutes\n"
                        f"Transcript chunks: {self._chunk_count}\n"
                        f"Output directory: {session_dir}",
                        title="Session Complete",
                        border_style="green",
                    )
                )
            else:
                console.print(
                    "[yellow]No speech detected during the session. No notes generated.[/yellow]"
                )

            # Stop llama-server
            self._server_manager.stop()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Summ-AI: fully local AI-powered meeting note-taker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  summ-ai                          Start capturing your default playback device
  summ-ai --list-sources           List available audio monitor sources
  summ-ai --source NAME            Use a specific PulseAudio source
  summ-ai --language es            Force Spanish transcription
  summ-ai --save-audio             Also save the recorded audio
  summ-ai --no-diarize             Disable speaker identification (faster)

How it works:
  Captures everything you hear through your speakers/headset by recording
  from the PulseAudio/PipeWire monitor source of your default audio output.
  Works with any app: Teams, Zoom, Discord, WhatsApp, browser, etc.

Environment variables (prefix SUMMAI_):
  SUMMAI_WHISPER_CLI_PATH           Path to whisper-cli binary
  SUMMAI_WHISPER_MODEL_PATH         Path to Whisper GGML model
  SUMMAI_LLAMA_SERVER_PATH          Path to llama-server binary
  SUMMAI_LLM_HF_REPO               HuggingFace model (e.g., unsloth/Qwen3.5-9B-GGUF:Q4_K_M)
  SUMMAI_DIARIZE                    Enable speaker identification (true/false, default: true)
  SUMMAI_OUTPUT_DIR                 Output directory for notes
        """,
    )

    parser.add_argument(
        "--source", "-s",
        type=str,
        default=None,
        help="PulseAudio source name to capture from (default: auto-detect default sink monitor)",
    )
    parser.add_argument(
        "--list-sources", "-l",
        action="store_true",
        help="List available audio monitor sources and exit",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Force transcription language (e.g., 'es', 'en'). Default: auto-detect",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="HuggingFace model repo:quant for llama-server (default: unsloth/Qwen3.5-9B-GGUF:Q4_K_M)",
    )
    parser.add_argument(
        "--output-dir", "-o", type=str, default=None, help="Output directory for notes"
    )
    parser.add_argument(
        "--save-audio", action="store_true", help="Save the full recording as WAV"
    )
    parser.add_argument(
        "--chunk-duration",
        type=int,
        default=None,
        help="Audio chunk duration in seconds (default: 30)",
    )
    parser.add_argument(
        "--no-diarize",
        action="store_true",
        help="Disable LLM-based speaker diarization (faster, but no speaker labels)",
    )

    args = parser.parse_args()

    # Build settings
    settings = Settings()

    if args.language:
        settings.whisper_language = args.language
    if args.llm_model:
        settings.llm_hf_repo = args.llm_model
    if args.output_dir:
        settings.output_dir = Path(args.output_dir)
    if args.save_audio:
        settings.save_audio = True
    if args.chunk_duration:
        settings.audio_chunk_duration = args.chunk_duration
    if args.no_diarize:
        settings.diarize = False

    # List sources mode
    if args.list_sources:
        audio = AudioCapture(settings)
        audio.list_monitor_sources()
        return

    # Run the session
    session = MeetingSession(settings)
    session.run(source=args.source)


if __name__ == "__main__":
    main()
