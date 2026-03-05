#!/usr/bin/env python3
"""Quick smoke test for ffmpeg-based audio capture.

Captures 5 seconds of system audio, reports RMS/peak, and verifies
that AudioCapture produces non-zero data from the queue.

Play some audio (YouTube, music, anything) through your headset/speakers
before running this.
"""

import sys
import time

import numpy as np

# Ensure we can import from the project
sys.path.insert(0, ".")

from notetaker.config import Settings
from notetaker.audio import AudioCapture, _get_default_sink_monitor, _get_active_sink_monitor, _list_monitor_sources


def main():
    print("=== Audio Capture Smoke Test ===\n")

    # 1. Check monitor source detection
    default_mon = _get_default_sink_monitor()
    active_mon = _get_active_sink_monitor()
    all_mons = _list_monitor_sources()
    print(f"Default sink monitor: {default_mon}")
    print(f"Active sink monitor:  {active_mon}")
    print(f"All monitors ({len(all_mons)}):")
    for name, desc in all_mons:
        print(f"  {desc}  [{name}]")
    print()

    # 2. Create AudioCapture with short chunk duration for fast test
    settings = Settings()
    # Override: use 5-second chunks so we get data quickly
    settings.audio_chunk_duration = 5
    # Use a very low threshold so we don't accidentally filter out quiet monitor audio
    settings.silence_threshold = 0.0001

    capture = AudioCapture(settings)
    print("Starting capture for ~6 seconds...")
    print("(Make sure some audio is playing through your speakers/headset!)\n")

    try:
        capture.start()
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return 1

    # Wait for audio to accumulate (5s chunk + 1s buffer)
    time.sleep(6)

    # 3. Check if we got any chunks
    chunks_received = 0
    all_rms = []
    all_peak = []

    while True:
        chunk = capture.get_chunk(timeout=0.1)
        if chunk is None:
            break
        chunks_received += 1
        rms = float(np.sqrt(np.mean(chunk**2)))
        peak = float(np.max(np.abs(chunk)))
        all_rms.append(rms)
        all_peak.append(peak)
        print(f"  Chunk {chunks_received}: {len(chunk)} samples, RMS={rms:.6f}, Peak={peak:.6f}")

    capture.stop()

    # 4. Report results
    print(f"\n=== Results ===")
    print(f"Chunks received:       {chunks_received}")
    print(f"Silent chunks skipped: {capture._silent_chunks_skipped}")

    if chunks_received > 0:
        avg_rms = np.mean(all_rms)
        max_peak = np.max(all_peak)
        print(f"Average RMS:           {avg_rms:.6f}")
        print(f"Max Peak:              {max_peak:.6f}")

        if avg_rms > 0.0005:
            print("\n[PASS] Audio capture is working! Non-zero audio detected.")
            return 0
        else:
            print("\n[WARN] Audio levels are extremely low. Check your audio output.")
            return 1
    else:
        # Check if everything was filtered as silence
        if capture._silent_chunks_skipped > 0:
            print(f"\n[WARN] Got {capture._silent_chunks_skipped} chunks but ALL were below silence threshold.")
            print("       The silence_threshold may be too high for monitor source levels.")
            return 1
        else:
            print("\n[FAIL] No audio data received at all!")
            print("       ffmpeg may have failed to capture from the monitor source.")
            # Check ffmpeg stderr
            if capture._ffmpeg_proc:
                stderr = capture._ffmpeg_proc.stderr.read() if capture._ffmpeg_proc.stderr else b""
                if stderr:
                    print(f"       ffmpeg stderr: {stderr.decode()}")
            return 1


if __name__ == "__main__":
    sys.exit(main())
