"""Smoke test for the LFM2-Audio python-source-plugin path.

Same shape as ``examples/python-source-plugin/consume.py`` — the manifest's
``plugins`` field handles loading. The only difference vs the echo demo
is that this plugin's per-venv provision pulls torch + liquid-audio +
transformers (~5–7 GB on first run), so the first ``create_streaming_session``
call can take several minutes.

We exercise the **text-input** TTS path on ``LFM2AudioNode`` (the same
class supports both audio→speech and text→speech), so this smoke test
doesn't require a WAV file. For an end-to-end audio→speech demo see
``examples/lfm2_audio_typed_rpc_chat.py`` against this manifest.

Prereqs:
  pip install ...     # standard remotemedia Python deps for the host

Run:
  cd examples/lfm2-audio-python-source-plugin
  PYTHONPATH=<repo>/clients/python REMOTEMEDIA_PYTHON_SRC=<repo>/clients/python python consume.py

Or just:
  ./run.sh
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import remotemedia.runtime as rt

log = logging.getLogger("lfm2_audio_source_plugin_consumer")

# Three short prompts that the audio model should be able to render
# quickly. The text path bypasses ASR — the model treats each string
# as the user's spoken turn and replies with interleaved text+audio.
PROMPTS = [
    "Say hello in one sentence.",
    "What time is it where you are?",
    "Tell me a one-line joke.",
]


async def main_async(manifest_path: Path) -> None:
    log.info("loading manifest: %s", manifest_path)
    manifest_json = manifest_path.read_text()

    log.info(
        "creating streaming session — this provisions the per-plugin uv "
        "venv on first run (downloads torch + liquid-audio + transformers, "
        "5–7 GB, several minutes the first time)…"
    )
    session = await rt.create_streaming_session(manifest_json)
    log.info("session_id = %s", session.session_id)

    for prompt in PROMPTS:
        log.info("sending prompt: %r", prompt)
        await session.send_input({"type": "text", "data": prompt})

        text_pieces: list[str] = []
        audio_chunks = 0
        while True:
            out = await session.recv_data()
            if out is None:
                break
            kind = out.get("type", "?") if isinstance(out, dict) else type(out).__name__
            if kind == "text":
                snippet = out.get("data", "") if isinstance(out, dict) else ""
                if snippet == "<|audio_end|>":
                    break
                if snippet == "<|text_end|>":
                    continue
                text_pieces.append(snippet)
            elif kind == "audio":
                audio_chunks += 1
            else:
                log.debug("unexpected output kind: %s", kind)

        text = "".join(text_pieces).strip()
        log.info("→ text: %r", text)
        log.info("→ audio chunks: %d", audio_chunks)

    await session.signal_input_complete()
    await session.close()
    log.info("clean shutdown")


def main() -> None:
    here = Path(__file__).parent.resolve()
    manifest_path = here / "manifest.json"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s | %(message)s",
    )

    if not manifest_path.exists():
        sys.exit(f"manifest not found: {manifest_path}")

    try:
        asyncio.run(main_async(manifest_path))
    except KeyboardInterrupt:
        print("\nbye.", file=sys.stderr)


if __name__ == "__main__":
    main()
