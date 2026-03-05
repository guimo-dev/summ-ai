#!/usr/bin/env python3
"""End-to-end integration test: audio capture → whisper transcription → LLM summarization.

This test:
1. Captures 15 seconds of system audio via ffmpeg
2. Sends it to whisper-cli for transcription
3. Starts llama-server and generates a summary
4. Prints everything so you can verify the pipeline

Requirements:
- Audio must be playing through your speakers/headset (e.g., a YouTube video with speech)
- whisper-cli and whisper model must be available
- llama-server must be available (model will auto-download if not cached)

Usage:
    .venv/bin/python test_e2e.py [--skip-llm]
"""

import sys
import time
import argparse

import numpy as np

sys.path.insert(0, ".")

from notetaker.config import Settings
from notetaker.audio import AudioCapture
from notetaker.transcriber import Transcriber
from notetaker.summarizer import Summarizer
from notetaker.main import LlamaServerManager


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM summarization step")
    parser.add_argument("--duration", type=int, default=15, help="Capture duration in seconds")
    args = parser.parse_args()

    settings = Settings()
    # Use shorter chunk for testing
    settings.audio_chunk_duration = args.duration

    print(f"=== End-to-End Integration Test ===\n")
    print(f"Capture duration: {args.duration}s")
    print(f"Skip LLM: {args.skip_llm}\n")

    # --- Step 1: Audio Capture ---
    print("--- Step 1: Audio Capture ---")
    capture = AudioCapture(settings)
    try:
        capture.start()
    except RuntimeError as e:
        print(f"FAIL: {e}")
        return 1

    print(f"Capturing {args.duration}s of audio...")
    time.sleep(args.duration + 2)  # extra buffer

    chunks = []
    while True:
        chunk = capture.get_chunk(timeout=0.1)
        if chunk is None:
            break
        chunks.append(chunk)

    capture.stop()

    if not chunks:
        print(f"\nFAIL: No audio chunks received!")
        print(f"  Silent chunks skipped: {capture._silent_chunks_skipped}")
        return 1

    audio = np.concatenate(chunks)
    rms = float(np.sqrt(np.mean(audio**2)))
    peak = float(np.max(np.abs(audio)))
    duration = len(audio) / settings.audio_sample_rate
    print(f"\nAudio captured: {duration:.1f}s, {len(chunks)} chunk(s)")
    print(f"  RMS={rms:.6f}, Peak={peak:.6f}")
    print(f"  Silent chunks skipped: {capture._silent_chunks_skipped}")
    print(f"  PASS\n")

    # --- Step 2: Whisper Transcription ---
    print("--- Step 2: Whisper Transcription ---")
    transcriber = Transcriber(settings)
    try:
        transcriber.load_model()
    except RuntimeError as e:
        print(f"FAIL: {e}")
        return 1

    result = transcriber.transcribe(audio)

    if not result.full_text.strip():
        print(f"  WARNING: Whisper produced empty transcription (silence or inaudible)")
        print(f"  Processing time: {result.processing_time:.1f}s")
        print(f"  Language detected: {result.language}")
        if args.skip_llm:
            return 0
        print(f"  Skipping LLM step since there's nothing to summarize.")
        return 0

    print(f"\n  Language: {result.language}")
    print(f"  Segments: {len(result.segments)}")
    print(f"  Processing time: {result.processing_time:.1f}s")
    print(f"  Speed: {result.duration / result.processing_time:.1f}x real-time")
    print(f"\n  Transcript:")
    print(f"  ---")
    for seg in result.segments:
        print(f"  {seg}")
    print(f"  ---")
    print(f"  PASS\n")

    if args.skip_llm:
        print("--- Step 3: LLM Summarization (SKIPPED) ---")
        print("\nAll tested steps passed!")
        return 0

    # --- Step 3: LLM Summarization ---
    print("--- Step 3: LLM Summarization ---")
    server_manager = LlamaServerManager(settings)
    summarizer = Summarizer(settings)

    # Check if server is already running
    if summarizer.check_server():
        print("  llama-server already running, reusing it.")
        server_started_by_us = False
    else:
        print("  Starting llama-server...")
        try:
            server_manager.start()
        except RuntimeError as e:
            print(f"FAIL: {e}")
            return 1

        print("  Waiting for model to load (this can take up to 60s on first run)...")
        if not summarizer.wait_for_server(timeout=120):
            print("FAIL: llama-server did not become ready in time")
            server_manager.stop()
            return 1
        server_started_by_us = True

    # Generate summary
    print("  Generating summary...")
    start = time.time()
    try:
        summary = summarizer.final_summary(result.full_text)
    except Exception as e:
        print(f"FAIL: {e}")
        if server_started_by_us:
            server_manager.stop()
        return 1

    elapsed = time.time() - start
    print(f"\n  Summary generated in {elapsed:.1f}s:")
    print(f"  ---")
    print(summary)
    print(f"  ---")
    print(f"  PASS\n")

    if server_started_by_us:
        server_manager.stop()

    print("=== All steps passed! ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
