"""Audio capture module - captures system audio via PulseAudio/PipeWire monitor source.

Uses ffmpeg to read from PulseAudio monitor sources, which is the most reliable
method on PipeWire systems (sounddevice/parecord/pw-record can return zeros from
monitor sources due to a PipeWire quirk, but ffmpeg works correctly).

This captures everything that plays through your speakers/headset, regardless of
which application produces it (Teams, Zoom, Discord, WhatsApp, browser, etc.).
"""

import queue
import subprocess
import threading
import time
import wave
from pathlib import Path

import numpy as np
from rich.console import Console

from notetaker.config import Settings

console = Console()


def _get_default_sink_monitor() -> str | None:
    """Get the monitor source name for the current default PulseAudio/PipeWire sink."""
    try:
        result = subprocess.run(
            ["pactl", "get-default-sink"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            sink_name = result.stdout.strip()
            if sink_name:
                return f"{sink_name}.monitor"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _get_active_sink_monitor() -> str | None:
    """Detect the monitor of the sink that currently has audio playing."""
    try:
        sinks_result = subprocess.run(
            ["pactl", "list", "sinks", "short"],
            capture_output=True, text=True, timeout=5,
        )
        inputs_result = subprocess.run(
            ["pactl", "list", "sink-inputs", "short"],
            capture_output=True, text=True, timeout=5,
        )
        sources_result = subprocess.run(
            ["pactl", "list", "sources", "short"],
            capture_output=True, text=True, timeout=5,
        )

        if sinks_result.returncode != 0 or inputs_result.returncode != 0:
            return None

        # Map sink index -> sink name
        sink_map: dict[str, str] = {}
        for line in sinks_result.stdout.strip().splitlines():
            parts = line.split("\t")
            if len(parts) >= 2:
                sink_map[parts[0]] = parts[1]

        # Count sink-inputs per sink index
        sink_counts: dict[str, int] = {}
        for line in inputs_result.stdout.strip().splitlines():
            parts = line.split("\t")
            if len(parts) >= 2:
                sink_index = parts[1]
                sink_counts[sink_index] = sink_counts.get(sink_index, 0) + 1

        if not sink_counts:
            return None

        # Pick the sink with the most active inputs
        active_sink_index = max(sink_counts.items(), key=lambda kv: kv[1])[0]
        sink_name = sink_map.get(active_sink_index)
        if not sink_name:
            return None

        monitor_name = f"{sink_name}.monitor"

        # Verify monitor exists
        if sources_result.returncode == 0:
            for line in sources_result.stdout.strip().splitlines():
                parts = line.split("\t")
                if len(parts) >= 2 and parts[1] == monitor_name:
                    return monitor_name
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _get_sink_descriptions() -> dict[str, str]:
    """Map sink names to their human-readable descriptions."""
    descriptions: dict[str, str] = {}
    try:
        result = subprocess.run(
            ["pactl", "list", "sinks"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            current_name = None
            for line in result.stdout.splitlines():
                stripped = line.strip()
                if stripped.startswith("Name:"):
                    current_name = stripped.split(":", 1)[1].strip()
                elif stripped.startswith("Description:") and current_name:
                    descriptions[current_name] = stripped.split(":", 1)[1].strip()
                    current_name = None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return descriptions


def _list_monitor_sources() -> list[tuple[str, str]]:
    """List all available PulseAudio/PipeWire monitor sources.

    Returns list of (source_name, description) tuples.
    """
    monitors = []
    sink_descriptions = _get_sink_descriptions()

    try:
        result = subprocess.run(
            ["pactl", "list", "sources", "short"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().splitlines():
                parts = line.split("\t")
                if len(parts) >= 2 and ".monitor" in parts[1]:
                    source_name = parts[1]
                    # Derive sink name from monitor source name
                    sink_name = source_name.removesuffix(".monitor")
                    description = sink_descriptions.get(sink_name, source_name)
                    monitors.append((source_name, description))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return monitors


class AudioCapture:
    """Captures system audio from a PulseAudio/PipeWire monitor source.

    Uses ffmpeg with the PulseAudio input device to read from monitor sources.
    This is the most reliable method on PipeWire systems.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.sample_rate = settings.audio_sample_rate
        self.channels = settings.audio_channels
        self.chunk_duration = settings.audio_chunk_duration
        self.silence_threshold = settings.silence_threshold

        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._recording = False
        self._ffmpeg_proc: subprocess.Popen | None = None
        self._reader_thread: threading.Thread | None = None
        self._all_audio: list[np.ndarray] = []  # for saving full recording
        self._source_name: str | None = None
        self._silent_chunks_skipped = 0
        self._total_bytes_read = 0
        self._reader_error: str | None = None

    def _setup_monitor_source(self, source: str | None = None) -> str:
        """Set up the PulseAudio monitor source for recording."""
        if source:
            self._source_name = source
            return source

        # Auto-detect: prefer the sink that is actively playing audio
        active_monitor = _get_active_sink_monitor()
        if active_monitor:
            self._source_name = active_monitor
            console.print(
                f"[green]Auto-detected active monitor source:[/green] {active_monitor}"
            )
            return active_monitor

        # Fallback: use the default sink's monitor
        monitor = _get_default_sink_monitor()
        if monitor:
            self._source_name = monitor
            console.print(
                f"[green]Auto-detected monitor source:[/green] {monitor}"
            )
            return monitor

        # Fallback: list available monitors and pick the first
        monitors = _list_monitor_sources()
        if monitors:
            self._source_name = monitors[0][0]
            console.print(
                f"[yellow]Using first available monitor:[/yellow] {monitors[0][0]}"
            )
            return monitors[0][0]

        raise RuntimeError(
            "No monitor source found. Make sure PulseAudio/PipeWire is running.\n"
            "Check available sources with: pactl list sources short"
        )

    def list_monitor_sources(self) -> None:
        """Print all available monitor sources with human-readable names."""
        monitors = _list_monitor_sources()
        default_monitor = _get_default_sink_monitor()
        active_monitor = _get_active_sink_monitor()

        if monitors:
            console.print("[bold]Available monitor sources:[/bold]\n")
            for name, desc in monitors:
                markers = []
                if default_monitor and name == default_monitor:
                    markers.append("[green]DEFAULT[/green]")
                if active_monitor and name == active_monitor:
                    markers.append("[cyan]ACTIVE[/cyan]")
                marker_str = f" [{', '.join(markers)}]" if markers else ""
                console.print(f"  [bold]{desc}[/bold]{marker_str}")
                console.print(f"    [dim]{name}[/dim]")
        else:
            console.print("[yellow]No monitor sources found.[/yellow]")

        console.print(
            "\n[dim]Usage: summ-ai --source <name>[/dim]"
        )

    def _reader_loop(self) -> None:
        """Background thread: reads raw PCM from ffmpeg stdout, chunks it, queues it."""
        assert self._ffmpeg_proc is not None
        stdout = self._ffmpeg_proc.stdout
        assert stdout is not None

        bytes_per_sample = 2  # s16le = 2 bytes
        chunk_samples = self.sample_rate * self.chunk_duration
        chunk_bytes = chunk_samples * bytes_per_sample
        leftover = b""

        while self._recording:
            # Read in ~500ms blocks to stay responsive
            block_bytes = self.sample_rate // 2 * bytes_per_sample
            try:
                data = stdout.read(block_bytes)
            except (ValueError, OSError) as e:
                self._reader_error = f"Read error: {e}"
                break
            if not data:
                # ffmpeg closed stdout -- check if it exited with error
                if self._ffmpeg_proc.poll() is not None:
                    stderr = ""
                    if self._ffmpeg_proc.stderr:
                        try:
                            stderr = self._ffmpeg_proc.stderr.read().decode(errors="replace").strip()
                        except Exception:
                            pass
                    rc = self._ffmpeg_proc.returncode
                    self._reader_error = (
                        f"ffmpeg exited with code {rc}"
                        + (f": {stderr}" if stderr else "")
                    )
                else:
                    self._reader_error = "ffmpeg stdout closed unexpectedly"
                break

            self._total_bytes_read += len(data)
            leftover += data

            # Accumulate for save_audio
            if self.settings.save_audio:
                samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                self._all_audio.append(samples)

            while len(leftover) >= chunk_bytes:
                chunk_raw = leftover[:chunk_bytes]
                leftover = leftover[chunk_bytes:]

                chunk = np.frombuffer(chunk_raw, dtype=np.int16).astype(np.float32) / 32768.0
                rms = float(np.sqrt(np.mean(chunk**2)))
                if rms > self.silence_threshold:
                    self.audio_queue.put(chunk)
                else:
                    self._silent_chunks_skipped += 1

        # Flush remaining audio (if at least 3s of data -- shorter chunks hallucinate)
        min_flush_bytes = self.sample_rate * 2 * 3  # 3 seconds minimum
        if leftover and len(leftover) >= min_flush_bytes:
            chunk = np.frombuffer(leftover, dtype=np.int16).astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(chunk**2)))
            if rms > self.silence_threshold:
                self.audio_queue.put(chunk)

    def start(self, source: str | None = None):
        """Start capturing system audio via ffmpeg.

        Args:
            source: PulseAudio source name (e.g., 'alsa_output.xxx.monitor'),
                    or None for auto-detect.
        """
        # Verify ffmpeg is available
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
        except FileNotFoundError:
            raise RuntimeError(
                "ffmpeg not found. Install it with your package manager.\n"
                "  Arch: sudo pacman -S ffmpeg\n"
                "  Ubuntu: sudo apt install ffmpeg"
            )

        monitor_source = self._setup_monitor_source(source)
        console.print(f"[green]Capturing audio from:[/green] {monitor_source}")

        # Start ffmpeg: read from PulseAudio monitor, output raw s16le PCM to stdout
        cmd = [
            "ffmpeg",
            "-loglevel", "error",
            "-f", "pulse",
            "-i", monitor_source,
            "-ar", str(self.sample_rate),
            "-ac", str(self.channels),
            "-f", "s16le",       # raw PCM, 16-bit signed little-endian
            "-acodec", "pcm_s16le",
            "pipe:1",            # output to stdout
        ]

        self._ffmpeg_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,  # own process group, immune to Ctrl+C
        )

        # Brief health check: give ffmpeg 0.5s to fail fast (bad source name, etc.)
        time.sleep(0.5)
        if self._ffmpeg_proc.poll() is not None:
            stderr = ""
            if self._ffmpeg_proc.stderr:
                try:
                    stderr = self._ffmpeg_proc.stderr.read().decode(errors="replace").strip()
                except Exception:
                    pass
            rc = self._ffmpeg_proc.returncode
            raise RuntimeError(
                f"ffmpeg exited immediately (code {rc}).\n"
                f"  Source: {monitor_source}\n"
                + (f"  Error: {stderr}\n" if stderr else "")
                + "  Check the source name with: summ-ai --list-sources"
            )

        self._recording = True
        self._all_audio = []
        self._silent_chunks_skipped = 0
        self._total_bytes_read = 0
        self._reader_error = None

        # Start background reader thread
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()

        console.print("[bold green]Audio capture started.[/bold green]")

    def stop(self) -> None:
        """Stop capturing audio and print diagnostics."""
        self._recording = False

        if self._ffmpeg_proc is not None:
            self._ffmpeg_proc.terminate()
            try:
                self._ffmpeg_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._ffmpeg_proc.kill()
                self._ffmpeg_proc.wait()
            self._ffmpeg_proc = None

        if self._reader_thread is not None:
            self._reader_thread.join(timeout=5)
            self._reader_thread = None

        console.print("[bold red]Audio capture stopped.[/bold red]")

        # Print diagnostics so the user can understand what happened
        total_seconds = self._total_bytes_read / (self.sample_rate * 2) if self._total_bytes_read else 0
        console.print(
            f"[dim]  Audio captured: {total_seconds:.1f}s raw data, "
            f"{self.audio_queue.qsize()} chunks queued, "
            f"{self._silent_chunks_skipped} silent chunks skipped[/dim]"
        )
        if self._reader_error:
            console.print(f"[bold red]  Audio reader error: {self._reader_error}[/bold red]")
        if self._total_bytes_read == 0 and not self._reader_error:
            console.print(
                "[bold yellow]  WARNING: ffmpeg produced no audio data.[/bold yellow]\n"
                "[yellow]  This usually means the monitor source is not receiving audio.\n"
                "  Check that audio is playing and try: summ-ai --list-sources[/yellow]"
            )

    def save_full_audio(self, path: Path) -> None:
        """Save the full recorded audio to a WAV file."""
        if self._all_audio:
            full = np.concatenate(self._all_audio)
            audio_int16 = (full * 32767).astype(np.int16)
            with wave.open(str(path), "wb") as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_int16.tobytes())
            console.print(f"[green]Full audio saved to:[/green] {path}")

    def get_chunk(self, timeout: float = 1.0) -> np.ndarray | None:
        """Get the next audio chunk from the queue.

        Returns None if no chunk is available within the timeout.
        """
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    @property
    def is_recording(self) -> bool:
        return self._recording
