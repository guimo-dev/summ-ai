"""Summarization module - uses llama-server (llama.cpp) OpenAI-compatible API with Qwen3.5.

Part of Summ-AI: fully local AI-powered meeting note-taker.
"""

import re
import time
from urllib.request import Request, urlopen
from urllib.error import URLError
import json as _json
from dataclasses import dataclass, field

from rich.console import Console

from notetaker.config import Settings


def _strip_think_tags(text: str) -> str:
    """Strip Qwen3.5's <think>...</think> reasoning blocks from the response."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()


console = Console()


# ---------------------------------------------------------------------------
# Speaker diarization prompt
# ---------------------------------------------------------------------------

DIARIZATION_SYSTEM_PROMPT = """\
You are a speaker diarization assistant. Your ONLY job is to add speaker labels \
to meeting transcript segments. You receive raw transcript text and must return \
the SAME text with speaker labels prepended to each speech turn.

RULES:
1. Use names when they can be inferred from context (e.g., someone says "Thanks, Maria" \
→ the next speaker is Maria). Otherwise use Speaker 1, Speaker 2, etc.
2. Preserve the EXACT wording of the transcript — do NOT rephrase, summarize, or add content.
3. A speaker change is indicated by a shift in topic, a greeting/response pattern, \
question-then-answer flow, or a change in language/style.
4. If you cannot determine a speaker change, keep the text under the current speaker.
5. Lines prefixed with [possible hallucination] should keep that prefix.
6. Output ONLY the labeled transcript. No explanations, no preamble."""

DIARIZATION_PROMPT = """\
Add speaker labels to this transcript segment. {context_instruction}

Transcript:
{transcript}"""


@dataclass
class SpeakerContext:
    """Tracks known speakers across transcript chunks for consistent labeling."""

    speakers: dict[str, str] = field(default_factory=dict)
    """Map of speaker label -> description/characteristics seen so far.
    Example: {"Maria": "project manager, speaks Spanish sometimes",
              "Speaker 1": "discusses backend code, male voice cues"}
    """
    last_speaker: str = ""
    """The last speaker identified in the previous chunk."""
    raw_history: list[str] = field(default_factory=list)
    """Last few lines of diarized text for context continuity (rolling window)."""

    MAX_HISTORY_LINES: int = 15  # Keep last N lines for context

    def update(self, diarized_text: str) -> None:
        """Update context from a newly diarized chunk."""
        lines = [ln for ln in diarized_text.strip().splitlines() if ln.strip()]
        if not lines:
            return

        # Extract speaker names from lines like "**Speaker 1:** text", "**Name**: text", or "Name: text"
        speaker_pattern = re.compile(r"^(?:\*\*(.+?)(?::\*\*|\*\*:)|\s*([^:*]+):)\s")
        for line in lines:
            m = speaker_pattern.match(line)
            if m:
                name = (m.group(1) or m.group(2)).strip()
                self.last_speaker = name
                if name not in self.speakers:
                    self.speakers[name] = ""

        # Keep rolling window of recent lines
        self.raw_history.extend(lines)
        self.raw_history = self.raw_history[-self.MAX_HISTORY_LINES:]

    def build_context_instruction(self) -> str:
        """Build a context instruction string for the diarization prompt."""
        parts = []

        if self.speakers:
            speaker_list = ", ".join(self.speakers.keys())
            parts.append(f"Speakers identified so far: {speaker_list}.")
        if self.last_speaker:
            parts.append(f"The last speaker in the previous segment was {self.last_speaker}.")
        if self.raw_history:
            recent = "\n".join(self.raw_history[-5:])
            parts.append(f"Recent context from previous segment:\n{recent}")

        if parts:
            return "\n".join(parts)
        return "This is the first segment of the meeting — no prior context."

INTERMEDIATE_SUMMARY_PROMPT = """\
Provide a concise intermediate summary of this portion of the meeting transcript. \
Focus on preserving:
1. Key discussion points and decisions
2. All technical terms and proper nouns exactly as mentioned
3. Action items or tasks assigned
4. Any speaker attributions

Keep the summary detailed enough that no important information is lost. \
Write in the same language as the transcript.

Transcript:
{transcript}
"""

FINAL_SUMMARY_PROMPT = """\
Below is the complete meeting transcript (and any intermediate summaries). \
Produce the final structured meeting notes following the format specified in your instructions.

Be thorough and precise. Include ALL technical details and specific \
references mentioned during the meeting. Do not omit any decisions or action items.

If the transcript contains speaker labels (e.g., "**Maria:**" or "**Speaker 1:**"), \
attribute statements and action items to the correct speakers in the notes. \
If no speaker labels are present, omit speaker attribution.

{content}
"""


class Summarizer:
    """Generates structured meeting notes using llama-server's OpenAI-compatible API."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._base_url = f"http://{settings.llm_host}:{settings.llm_port}"
        self._system_prompt = settings.system_prompt
        self._intermediate_summaries: list[str] = []
        self._speaker_ctx = SpeakerContext()

    def check_server(self) -> bool:
        """Check if llama-server is running and healthy."""
        try:
            req = Request(f"{self._base_url}/health")
            with urlopen(req, timeout=5) as resp:
                data = _json.loads(resp.read())
                status = data.get("status", "")
                if status == "ok":
                    console.print(f"[green]llama-server is ready[/green]")
                    return True
                elif status == "loading model":
                    console.print(
                        f"[yellow]llama-server is loading the model, please wait...[/yellow]"
                    )
                    return False
                else:
                    console.print(f"[yellow]llama-server status: {status}[/yellow]")
                    return False
        except (URLError, OSError, TimeoutError):
            return False

    def wait_for_server(self, timeout: float = 300.0) -> bool:
        """Wait for llama-server to become ready (model loaded).

        Args:
            timeout: Maximum seconds to wait.

        Returns:
            True if server is ready, False if timed out.
        """
        start = time.time()
        while time.time() - start < timeout:
            if self.check_server():
                return True
            time.sleep(2)
        return False

    def _chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        presence_penalty: float = 1.5,
        max_tokens: int = 8192,
    ) -> str:
        """Call llama-server's /v1/chat/completions endpoint.

        Uses Qwen3.5 recommended sampling params for non-thinking mode.
        """
        payload = _json.dumps({
            "model": "qwen3.5",  # llama-server ignores this but it's required by the API
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "presence_penalty": presence_penalty,
            "max_tokens": max_tokens,
        }).encode("utf-8")

        req = Request(
            f"{self._base_url}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urlopen(req, timeout=300) as resp:
            data = _json.loads(resp.read())

        content = data["choices"][0]["message"]["content"]
        return _strip_think_tags(content)

    def intermediate_summary(self, transcript_chunk: str) -> str:
        """Generate an intermediate summary for a chunk of transcript.

        This is called periodically during the meeting to keep context manageable.
        """
        console.print("[cyan]Generating intermediate summary...[/cyan]")
        start = time.time()

        summary = self._chat_completion(
            messages=[
                {"role": "system", "content": self._system_prompt},
                {
                    "role": "user",
                    "content": INTERMEDIATE_SUMMARY_PROMPT.format(
                        transcript=transcript_chunk
                    ),
                },
            ],
            temperature=0.3,
            top_p=0.9,
        )

        self._intermediate_summaries.append(summary)

        elapsed = time.time() - start
        console.print(f"[dim]Intermediate summary generated in {elapsed:.1f}s[/dim]")
        return summary

    def final_summary(self, full_transcript: str) -> str:
        """Generate the final structured meeting notes.

        Combines intermediate summaries (if any) with the full transcript context
        to produce comprehensive notes.
        """
        console.print("[bold cyan]Generating final meeting notes...[/bold cyan]")
        start = time.time()

        # Build the content for final summarization
        content_parts = []

        if self._intermediate_summaries:
            content_parts.append("## Intermediate Summaries from the Meeting\n")
            for i, summary in enumerate(self._intermediate_summaries, 1):
                content_parts.append(f"### Part {i}\n{summary}\n")
            content_parts.append("\n## Full Transcript (for verification)\n")

        # If transcript is very long, we include intermediate summaries + last portion
        max_transcript_chars = 12000
        if len(full_transcript) > max_transcript_chars and self._intermediate_summaries:
            content_parts.append(
                f"[Transcript truncated - see intermediate summaries above for earlier content]\n\n"
                f"...{full_transcript[-max_transcript_chars:]}"
            )
        else:
            content_parts.append(full_transcript)

        content = "\n".join(content_parts)

        notes = self._chat_completion(
            messages=[
                {"role": "system", "content": self._system_prompt},
                {
                    "role": "user",
                    "content": FINAL_SUMMARY_PROMPT.format(content=content),
                },
            ],
            temperature=0.2,
            top_p=0.9,
        )

        elapsed = time.time() - start
        console.print(f"[green]Final notes generated in {elapsed:.1f}s[/green]")
        return notes

    def diarize_transcript(self, transcript_text: str) -> str:
        """Add speaker labels to a transcript chunk using the LLM.

        Uses the running speaker context to maintain consistent speaker labels
        across chunks. The transcript text is returned with speaker labels
        prepended to each speech turn (e.g., "**Maria:** ...", "**Speaker 1:** ...").

        Args:
            transcript_text: Raw transcript text from a single chunk.

        Returns:
            The same text with speaker labels added.
        """
        context_instruction = self._speaker_ctx.build_context_instruction()

        prompt = DIARIZATION_PROMPT.format(
            context_instruction=context_instruction,
            transcript=transcript_text,
        )

        # Use lower max_tokens — output is roughly same length as input, plus labels.
        # Estimate: input chars * 1.5 tokens/char is generous; cap at 2048.
        estimated_tokens = min(max(len(transcript_text) * 2, 512), 2048)

        try:
            diarized = self._chat_completion(
                messages=[
                    {"role": "system", "content": DIARIZATION_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                top_p=0.9,
                top_k=20,
                presence_penalty=1.0,
                max_tokens=estimated_tokens,
            )
        except Exception as e:
            console.print(f"[yellow]Diarization failed, using raw transcript: {e}[/yellow]")
            return transcript_text

        # Sanity check: if the LLM returned something drastically different in length
        # (could indicate hallucination/summarization instead of labeling), fall back
        if len(diarized) < len(transcript_text) * 0.4:
            console.print(
                "[yellow]Diarization output too short (possible summarization), "
                "using raw transcript[/yellow]"
            )
            return transcript_text

        # Update the running speaker context with the new labels
        self._speaker_ctx.update(diarized)

        return diarized

    @property
    def speaker_context(self) -> SpeakerContext:
        """Access the current speaker context (read-only)."""
        return self._speaker_ctx

    def reset(self):
        """Reset intermediate summaries and speaker context for a new meeting."""
        self._intermediate_summaries = []
        self._speaker_ctx = SpeakerContext()
