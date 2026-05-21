# /// script
# dependencies = [
#   # Windows + Python 3.12: pin torch/torchaudio to the CUDA 12.8 build.
#   # The PyPI default `torch` resolves to the CPU-only wheel on Windows,
#   # which makes LFM2 init crash on systems with NVIDIA hardware
#   # ("Torch not compiled with CUDA enabled"). Marker-gated so non-Windows /
#   # non-cp312 installs fall back to @python_requires's `torch>=2.1` and
#   # whatever the platform's default index serves (Linux PyPI torch
#   # already ships CUDA wheels via manylinux).
#   "torch @ https://download.pytorch.org/whl/cu128/torch-2.11.0%2Bcu128-cp312-cp312-win_amd64.whl ; sys_platform == 'win32' and python_version == '3.12'",
#   "torchaudio @ https://download.pytorch.org/whl/cu128/torchaudio-2.11.0%2Bcu128-cp312-cp312-win_amd64.whl ; sys_platform == 'win32' and python_version == '3.12'",
# ]
# ///

"""
VENDORED from `clients/python/remotemedia/nodes/ml/lfm2_audio.py`
for the `examples/lfm2-audio-python-source-plugin/` source-plugin path.

This file is intentionally kept byte-compatible with the in-tree node
so behaviour matches drop-in. When the upstream changes, re-copy:

    cp clients/python/remotemedia/nodes/ml/lfm2_audio.py \\
       examples/lfm2-audio-python-source-plugin/lfm2_audio_source.py

The two-line _NODE_REGISTRY alias at the bottom of this file is the
only divergence — see `examples/python-source-plugin/echo_python.py`
for the canonical pattern.

----------------------------------------------------------------------

LFM2-Audio speech-to-speech node — multiprocess-capable, control-bus-aware.

Speech-to-speech conversational AI built on Liquid AI's LFM2-Audio-1.5B.
Accepts audio on the main input, generates interleaved text and audio on
the output. The same aux-port control surface as :mod:`lfm2_text` is
wired up here so a gRPC/WebRTC client can steer a live voice agent
between turns without tearing the session down.

## Control-bus surface

Publishes to these aux ports arrive here as
``RuntimeData.Json({"__aux_port__": <port>, "payload": {...}})``:

    audio.in.context        → store RAG / retrieval text used on the
                              next turn; invalidates cached chat state so
                              the new system turn includes the context.
    audio.in.system_prompt  → replace the persona / behaviour prompt;
                              cached chat states are dropped.
    audio.in.reset          → drop conversation history for all sessions.
    audio.in.barge_in       → request immediate cancellation of the
                              currently-generating turn (if any). The
                              generator loop checks the flag on every
                              token batch and bails out cleanly.
    audio.in                → main channel: one audio utterance, treated
                              as the user turn.

The main audio channel is still validated as ``RuntimeData.Audio`` — the
aux-port envelope always arrives as text-wrapped JSON, so it's peeled
off BEFORE audio validation runs.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional, TYPE_CHECKING, Union

# Heavy ML deps are optional so the module can be imported for node
# registration even when liquid_audio / torch aren't installed. Capture
# the actual failure reason — the previous blanket `except ImportError`
# silently masked "liquid_audio is installed but import ChatState
# failed" cases, producing misleading "please pip install liquid-audio"
# errors even when the venv had the package.
_ML_IMPORT_ERROR: Optional[BaseException] = None
try:
    import numpy as np
    import torch
    import torchaudio  # noqa: F401
    from liquid_audio import ChatState, LFMModality
    from liquid_audio import LFM2AudioModel, LFM2AudioProcessor
    _ML_DEPS_AVAILABLE = True
except BaseException as _exc:
    _ML_DEPS_AVAILABLE = False
    _ML_IMPORT_ERROR = _exc
    np = None  # type: ignore
    torch = None  # type: ignore
    ChatState = None  # type: ignore
    LFMModality = None  # type: ignore
    LFM2AudioModel = None  # type: ignore
    LFM2AudioProcessor = None  # type: ignore
    logging.getLogger(__name__).warning(
        "LFM2AudioNode ML imports failed (%s): %s",
        type(_exc).__name__, _exc,
    )

if _ML_DEPS_AVAILABLE:
    try:  # torch dynamo is opportunistic — don't let it break inference
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True
    except (ImportError, AttributeError):
        pass

if TYPE_CHECKING:
    from remotemedia.core.multiprocessing.data import RuntimeData

try:
    from remotemedia.core.multiprocessing.data import RuntimeData
    _HAS_RUNTIME_DATA = True
except ImportError:
    _HAS_RUNTIME_DATA = False
    RuntimeData = None  # type: ignore
    logging.warning(
        "[LFM2AudioNode] RuntimeData bindings not available. "
        "Using fallback implementation."
    )
    
try:
    from remotemedia.core.multiprocessing.data import (
        numpy_to_audio,
        audio_to_numpy,  # noqa: F401  (exported for downstream users)
    )
except ImportError:
    numpy_to_audio = None  # type: ignore
    audio_to_numpy = None  # type: ignore

from remotemedia.core.multiprocessing import (
    MultiprocessNode,
    NodeConfig,
    python_requires,
    register_node,
)
from remotemedia.rpc import rpc

logger = logging.getLogger(__name__)


DEFAULT_SYSTEM_PROMPT = "Respond with interleaved text and audio."
AUX_PORT_KEY = "__aux_port__"


@dataclass
class ConversationState:
    """One live ``liquid_audio.ChatState`` plus session bookkeeping."""

    session_id: str
    chat_state: Any  # liquid_audio.ChatState
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    turn_count: int = 0

    def touch(self) -> None:
        self.last_accessed = datetime.now()


@register_node("LFM2AudioNode")
@python_requires(
    [
        # LFM2-Audio pulls ChatState/LFM2AudioModel/LFM2AudioProcessor from
        # the liquid_audio SDK, which itself has torch/torchaudio/transformers
        # as transitive deps. We pin transformers to the LFM2-compatible
        # release line used by the text sibling.
        "liquid-audio>=0.1",
        # See control_bus_test_server.rs for the full reasoning. In short:
        # liquid_audio 1.1.0 imports a transformers-4.54-era private
        # symbol (`Lfm2HybridConvCache`) that 5.x removed.
        "transformers>=4.54.0,<5.0",
        "torch>=2.1",
        "torchaudio>=2.1",
        "accelerate>=0.33",
        # safetensors >=0.5 on Windows crashes with an access violation
        # inside `safe_open(...).get_tensor()` while accelerate's
        # `load_checkpoint_in_model` mmaps the .safetensors shards
        # (verified on safetensors 0.7.0, torch 2.11.0+cu128, accelerate
        # 1.13.0, RTX 4090). The crash is reproducible from a 5-line
        # `LFM2AudioModel.from_pretrained(...)` script — exit code 5,
        # access violation in `torch.storage.UntypedStorage.__getitem__`.
        # 0.4.5 uses the older non-mmap GPU path and is stable. liquid_audio
        # 1.2.0's `accelerate>=1.10.1` constraint allows this freely
        # because nothing in that stack pins safetensors itself.
        "safetensors<0.5",
    ]
)
class LFM2AudioNode(MultiprocessNode):
    """
    Multi-turn speech-to-speech node with control-bus aux ports.

    Main channel consumes ``RuntimeData.Audio`` (24 kHz mono float32)
    and yields interleaved ``RuntimeData.Text`` and ``RuntimeData.Audio``.
    Aux ports (context / system_prompt / reset / barge_in) are handled
    out-of-band — aux messages produce no outputs of their own.
    """

    # ────── Construction (dual-mode: in-process kwargs OR NodeConfig) ──

    def __init__(
        self,
        config: Union[NodeConfig, Dict[str, Any], None] = None,
        *,
        node_id: Optional[str] = None,
        name: Optional[str] = None,
        hf_repo: str = "LiquidAI/LFM2.5-Audio-1.5B",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        device: Optional[str] = None,
        audio_temperature: float = 1.0,
        audio_top_k: int = 4,
        max_new_tokens: int = 4096,
        sample_rate: int = 24000,
        session_timeout_minutes: int = 30,
        text_only: bool = False,
        audio_batch_size: int = 12,
        first_chunk_audio_batch_size: Optional[int] = None,
        warmup_on_init: bool = True,
        **kwargs: Any,
    ) -> None:
        # Multiprocess runner's 3-attempt construction: str → TypeError → config=...
        # Reject the bare string form so we land on the config path where
        # manifest params actually reach us.
        if isinstance(config, str):
            raise TypeError(
                "LFM2AudioNode requires NodeConfig or keyword-only params; "
                "bare positional node_id not supported"
            )
        if config is None:
            config = NodeConfig(
                node_id=node_id or name or "lfm2_audio",
                node_type="LFM2AudioNode",
                params={},
            )
        elif isinstance(config, dict):
            config = NodeConfig(
                node_id=config.get("node_id", node_id or "lfm2_audio"),
                node_type=config.get("node_type", "LFM2AudioNode"),
                params=config.get("params", {}),
            )

        super().__init__(config, **kwargs)

        params = config.params or {}
        self.hf_repo = params.get("hf_repo", hf_repo)
        self._system_prompt = params.get("system_prompt", system_prompt)
        self.audio_temperature = float(params.get("audio_temperature", audio_temperature))
        self.audio_top_k = int(params.get("audio_top_k", audio_top_k))
        self.max_new_tokens = int(params.get("max_new_tokens", max_new_tokens))
        self.sample_rate = int(params.get("sample_rate", sample_rate))
        self.session_timeout_minutes = int(
            params.get("session_timeout_minutes", session_timeout_minutes)
        )
        self.text_only = bool(params.get("text_only", text_only))
        # How many confirmed audio tokens to accumulate before emitting a
        # Mimi-decoded audio chunk.
        #
        # Naively lowering this reduces TTFA (fewer decode iters before
        # first emit) but PROPORTIONALLY increases Mimi codec calls and
        # IPC sends for the rest of the response. Empirically with
        # batch_size=4 the per-call Mimi overhead dominated the savings
        # and total TTFA *went up*. Keep this at 12 for steady-state and
        # use `first_chunk_audio_batch_size` to cut just the first chunk.
        self.audio_batch_size = max(1, int(params.get("audio_batch_size", audio_batch_size)))
        # Optional override for the *first* emitted chunk per turn. When
        # set (and < `audio_batch_size`), we emit the first chunk as soon
        # as this many audio tokens are confirmed — getting audio out the
        # door faster — then revert to the normal `audio_batch_size` for
        # the rest of the response. Lets us trade ~minimum-perceivable
        # first-chunk duration for TTFA without paying the steady-state
        # overhead.
        _fc = params.get("first_chunk_audio_batch_size", first_chunk_audio_batch_size)
        self.first_chunk_audio_batch_size = (
            max(1, int(_fc)) if _fc is not None else None
        )
        # CUDA-kernel JIT warmup. `LFM2AudioModel.from_pretrained` loads
        # weights into GPU memory but does NOT compile the kernels —
        # PyTorch defers that to first use. Without a warmup pass, the
        # user's first peer.offer pays ~3-5 s of kernel compilation +
        # KV-cache allocation + first-time audio-encoder/Mimi-codec
        # forward inside `initialize()`'s caller (typically
        # `WarmSessionPool::prewarm`), the cost is paid at server boot
        # instead of on the first peer connection.
        #
        # Set `warmup_on_init=False` (or `"warmup_on_init": false` in
        # manifest params) to skip — useful in tests that don't need
        # the steady-state latency floor.
        self.warmup_on_init = bool(params.get("warmup_on_init", warmup_on_init))

        # Pin to a *specific* CUDA index. liquid_audio 1.1.0 with a bare
        # `device="cuda"` lets the library / HF accelerate spread submodules
        # across cuda:0/cuda:1/CPU on multi-GPU hosts (some weights get
        # CPU-offloaded), which then crashes during prefill with:
        #   "Expected all tensors to be on the same device, but got index
        #    is on cuda:1, different from other tensors on cpu"
        req_device = params.get("device", device)
        if req_device is None:
            if _ML_DEPS_AVAILABLE and torch.cuda.is_available():
                self.device = "cuda:0"
            else:
                self.device = "cpu"
        elif req_device == "cuda":
            self.device = "cuda:0"
        else:
            self.device = req_device

        self._processor: Any = None
        self._model: Any = None
        self._initialized = False

        # Per-session chat state. Keyed by logical session id (which is
        # the IPC session id in the multiprocess runner).
        self._sessions: Dict[str, ConversationState] = {}
        self._cleanup_task: Optional[asyncio.Task] = None

        # Aux-port state.
        self._context: str = ""          # retrieval/RAG block
        self._interrupt: bool = False    # barge-in latch (consumed by token loop)

        # Friendly name used by Pipeline.get_node(name) for in-process use.
        self.name = name or config.node_id

        self.is_streaming = True
        logger.info(
            "LFM2AudioNode initialized: device=%s text_only=%s",
            self.device, self.text_only,
        )

    # ────── In-process control-plane-analog API ───────────────────────
    #
    # These setters are used by in-process tests (`pl.get_node("lfm").set_context(...)`)
    # and by `_handle_aux_port` when the same node runs in a multiprocess
    # worker driven by the Session Control Bus. Keeping a single surface
    # ensures both paths behave identically.

    def set_context(self, docs: str) -> None:
        """Replace the retrieval/RAG context applied to the system turn."""
        self._context = docs or ""
        self._invalidate_sessions("context")

    def clear_context(self) -> None:
        self._context = ""
        self._invalidate_sessions("context-clear")

    def set_system_prompt(self, prompt: str) -> None:
        """Replace the persona/system prompt. Drops cached chat states."""
        self._system_prompt = prompt or DEFAULT_SYSTEM_PROMPT
        self._invalidate_sessions("system-prompt")

    def reset_history(self) -> None:
        """Drop conversation history. Next turn starts from the system prompt."""
        self._invalidate_sessions("reset")

    def request_barge_in(self) -> None:
        """Signal the active generation loop to stop ASAP. Cleared next turn."""
        self._interrupt = True
        logger.info("[%s] barge-in requested", self.node_id)

    def _invalidate_sessions(self, reason: str) -> None:
        if self._sessions:
            logger.info(
                "[%s] dropping %d cached chat state(s) (%s)",
                self.node_id, len(self._sessions), reason,
            )
            self._sessions.clear()

    # ────── Typed-RPC surface (dispatched via @rpc on the control bus) ─
    #
    # These async methods mirror the sync in-process API above and are
    # dispatched by the runner's `_dispatch_aux_frame` when a typed-RPC
    # aux-port frame arrives on `<node>.in.<method_name>`.
    #
    # The sync setters are kept for in-process callers (tests, Pipeline API).
    # Legacy raw-publish callers (``ctrl.publish("audio.in.context", ...)``)
    # continue to work via the `_handle_aux_port` / `process()` fall-through
    # (the dispatcher returns False for legacy-shaped frames even when the port
    # name matches a registered @rpc method).

    @rpc
    async def set_context(self, context: str) -> None:  # type: ignore[override]
        """Replace the RAG/retrieval context for the next turn."""
        self._context = context or ""
        self._invalidate_sessions("context")

    @rpc
    async def clear_context(self) -> None:  # type: ignore[override]
        """Clear the RAG/retrieval context."""
        self._context = ""
        self._invalidate_sessions("context-clear")

    @rpc
    async def set_system_prompt(self, prompt: str) -> None:  # type: ignore[override]
        """Replace the persona/system prompt. Drops cached chat states."""
        self._system_prompt = prompt or DEFAULT_SYSTEM_PROMPT
        self._invalidate_sessions("system-prompt")

    @rpc
    async def reset_history(self) -> None:  # type: ignore[override]
        """Drop conversation history. Next turn starts from the system prompt."""
        self._invalidate_sessions("reset")

    @rpc
    async def barge_in(self) -> None:
        """Signal the active generation loop to stop ASAP. Cleared next turn."""
        self._interrupt = True
        logger.info("[%s] barge-in requested (typed RPC)", self.node_id)

    # ────── MultiprocessNode contract ─────────────────────────────────

    async def initialize(self) -> None:
        if not _ML_DEPS_AVAILABLE:
            # Surface the actual import failure instead of the
            # misleading "please pip install" error. Common cases:
            # - `from liquid_audio import ChatState` fails because the
            #   API moved between versions.
            # - torchaudio is imported but the native ffmpeg backend
            #   isn't available on the host (OSError from dlopen).
            cause = _ML_IMPORT_ERROR
            detail = (
                f"{type(cause).__name__}: {cause}" if cause is not None
                else "unknown import failure"
            )
            raise RuntimeError(
                f"LFM2AudioNode ML stack failed to import — {detail}. "
                "Required packages: liquid_audio, torch, torchaudio."
            ) from cause
        if self._initialized:
            return

        logger.info(
            "Initializing LFM2-Audio from %r on %s", self.hf_repo, self.device
        )
        if self.device == "cpu" and not torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

        # liquid_audio's from_pretrained() takes the local-dir branch
        # only when given a pathlib.Path; a str always goes through
        # snapshot_download(), which fails the repo-id validator on
        # filesystem paths. Auto-detect: if the value resolves to an
        # existing directory, hand it in as a Path. Otherwise pass the
        # original string through and let HF resolve it as a repo id.
        from pathlib import Path as _Path
        _maybe_local = _Path(str(self.hf_repo))
        load_target = _maybe_local if _maybe_local.is_dir() else self.hf_repo

        try:
            self._processor = LFM2AudioProcessor.from_pretrained(
                load_target, device=self.device
            )
        except TypeError:
            self._processor = LFM2AudioProcessor.from_pretrained(load_target)
            if self.device == "cpu":
                self._processor = self._processor.to("cpu")
        self._processor = self._processor.eval()

        # liquid_audio 1.1.0 changed `LFM2AudioModel.from_pretrained`:
        #   - dropped the `attn_implementation=` kwarg.
        #   - defaults to device="cuda", dtype=torch.bfloat16 regardless
        #     of host capability; we must pass our own device down so
        #     the inner model isn't constructed on CUDA on a CPU-only
        #     box (which aborts with "Torch not compiled with CUDA
        #     enabled" the moment any tensor is allocated).
        #
        # Pass `device=` / `dtype=` explicitly when the API accepts them
        # and fall back to post-construct `.to(device)` otherwise — old
        # 0.x liquid_audio didn't accept device/dtype on from_pretrained.
        #
        # Windows: avoid passing a CUDA device into from_pretrained. The
        # accelerate `load_checkpoint_in_model` path then calls
        # `safetensors.torch.load_file(..., device="cuda:N")`, which
        # tries to materialise tensors directly into GPU memory and
        # reliably crashes with a Windows fatal access violation
        # inside `torch.storage.UntypedStorage.__getitem__` (verified on
        # safetensors 0.4.5 + safetensors 0.7.0, torch 2.11.0+cu128 + 2.12.0,
        # accelerate 1.13.0, RTX 4090). Load on CPU and let the explicit
        # `.to(self.device)` below consolidate onto the target device —
        # one extra host->device copy, but stable.
        import sys as _sys
        _on_windows = _sys.platform == "win32"

        model_kwargs: Dict[str, Any] = {}
        if self.device.startswith("cuda"):
            if _on_windows:
                model_kwargs["device"] = "cpu"
            else:
                model_kwargs["device"] = self.device
            model_kwargs["dtype"] = torch.bfloat16
        else:
            model_kwargs["device"] = "cpu"
            model_kwargs["dtype"] = torch.float32

        try:
            self._model = LFM2AudioModel.from_pretrained(load_target, **model_kwargs)
        except TypeError:
            # Older liquid_audio that doesn't accept device / dtype —
            # load on CPU and move afterwards.
            self._model = LFM2AudioModel.from_pretrained(load_target)

        # Defensively consolidate every submodule onto our target device.
        # The `device=` kwarg above isn't always enough — liquid_audio 1.1.0
        # internally invokes HF accelerate paths that can offload embeddings
        # to CPU (cuda:0/cuda:1/cpu spread). Calling `.to(self.device)` on
        # the whole model forces a single-device layout and avoids the
        # device-mismatch RuntimeError on the first prefill.
        try:
            self._model = self._model.to(self.device)
        except (RuntimeError, NotImplementedError) as e:
            logger.warning(
                "Could not consolidate model onto %s (%s); relying on "
                "library device placement",
                self.device, e,
            )
        self._model = self._model.eval()

        self._initialized = True
        logger.info("LFM2-Audio model loaded")

        # Run the warmup *before* spawning the session-cleanup task —
        # `initialize()` is awaited synchronously by
        # `WarmSessionPool::prewarm`, so any work here gates the
        # "READY" log on the server. The warmup compiles CUDA kernels
        # + allocates the first KV cache + exercises the Mimi codec
        # once, removing the 3-5 s cliff from the user's first turn.
        if self.warmup_on_init:
            self._warmup_inference()

        self._cleanup_task = asyncio.create_task(self._cleanup_expired_sessions())

    def _warmup_inference(self) -> None:
        """One-shot dummy forward pass that JIT-compiles CUDA kernels.

        Without this, the first real user turn pays ~3-5 s of kernel
        compilation + KV-cache allocation + first audio-encoder /
        Mimi-codec pass. Loading weights into GPU memory via
        `LFM2AudioModel.from_pretrained` does NOT eagerly compile —
        PyTorch defers until first use. Calling a tiny synthetic
        inference here ties that cost to plugin init, which
        `WarmSessionPool::prewarm` already awaits.

        Failures are caught and logged — a broken warmup must not
        take down `initialize()`. Worst case the user pays the
        cold-start on their first turn (the prior behaviour).
        """
        if self._model is None or self._processor is None:
            return

        t0 = time.perf_counter()
        try:
            # 0.25 s of silence at the model's native sample rate is
            # enough to exercise the audio encoder, the depth-former
            # path, and the Mimi codec — i.e. every CUDA kernel the
            # first real turn will hit. Smaller buffers risk
            # short-circuit paths skipping a kernel; longer adds
            # boot-time latency for no extra coverage.
            n_samples = max(1, int(self.sample_rate * 0.25))
            silence = np.zeros(n_samples, dtype=np.float32)
            wav = torch.from_numpy(silence).float().unsqueeze(0)

            chat = ChatState(self._processor)
            chat.new_turn("system")
            chat.add_text(self._system_prompt)
            chat.end_turn()
            chat.new_turn("user")
            chat.add_audio(wav, self.sample_rate)
            chat.end_turn()
            chat.new_turn("assistant")

            # `max_new_tokens=2` covers both the first-token kernel
            # (prefill) and the continuation kernel (decode step) —
            # JIT-compiling both. Discarding the output; this
            # `ChatState` is throwaway and never enters
            # `self._sessions`.
            gen = self._model.generate_interleaved(
                **chat,
                max_new_tokens=2,
                audio_temperature=self.audio_temperature,
                audio_top_k=self.audio_top_k,
            )
            for _ in gen:
                pass

            logger.info(
                "[%s] warmup inference complete in %.2fs",
                self.node_id, time.perf_counter() - t0,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[%s] warmup inference failed after %.2fs (%s) — "
                "first real turn will pay the kernel-compile cost",
                self.node_id, time.perf_counter() - t0, exc,
            )

    async def cleanup(self) -> None:
        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        self._model = None
        self._processor = None
        self._initialized = False
        self._sessions.clear()
        logger.info("LFM2-Audio model cleaned up")

    async def _cleanup_expired_sessions(self) -> None:
        while True:
            try:
                await asyncio.sleep(60)
                now = datetime.now()
                expired = [
                    sid for sid, s in self._sessions.items()
                    if (now - s.last_accessed).total_seconds() / 60 > self.session_timeout_minutes
                ]
                for sid in expired:
                    logger.info("Removing expired session: %s", sid)
                    self._sessions.pop(sid, None)
            except asyncio.CancelledError:
                break
            except Exception as exc:  # noqa: BLE001 — keep the cleaner alive
                logger.error("Session cleanup error: %s", exc)

    def _build_system_turn_text(self) -> str:
        if not self._context:
            return self._system_prompt
        return (
            f"{self._system_prompt}\n\n"
            f"Known facts you must use when relevant:\n{self._context}"
        )

    async def _get_or_create_session(self, session_id: str) -> ConversationState:
        if session_id in self._sessions:
            self._sessions[session_id].touch()
            return self._sessions[session_id]

        if self._processor is None:
            raise RuntimeError("LFM2AudioNode: initialize() must be called first")

        logger.info("Creating new conversation session: %s", session_id)
        chat = ChatState(self._processor)
        chat.new_turn("system")
        chat.add_text(self._build_system_turn_text())
        chat.end_turn()

        self._sessions[session_id] = ConversationState(
            session_id=session_id, chat_state=chat
        )
        return self._sessions[session_id]

    # ────── Aux-port envelope handling ────────────────────────────────

    def _extract_envelope(self, data: Any) -> Optional[tuple]:
        """
        Return ``(port_name, payload_dict)`` if ``data`` is an aux-port
        envelope, else ``None``.

        Accepts dict / JSON string / RuntimeData.Text-of-JSON.
        """
        blob = self._to_dict(data)
        if not isinstance(blob, dict):
            return None
        port = blob.get(AUX_PORT_KEY)
        if not isinstance(port, str) or not port:
            return None
        payload = blob.get("payload")
        if not isinstance(payload, dict):
            payload = {"text": str(payload)} if payload is not None else {}
        return port, payload

    def _to_dict(self, data: Any) -> Any:
        if isinstance(data, dict):
            return data
        if isinstance(data, str):
            stripped = data.strip()
            if stripped.startswith("{"):
                try:
                    return json.loads(stripped)
                except json.JSONDecodeError:
                    return None
            return None
        if _HAS_RUNTIME_DATA and RuntimeData is not None and isinstance(data, RuntimeData):
            try:
                if data.is_text():
                    return self._to_dict(data.as_text())
            except Exception:  # noqa: BLE001
                return None
        return None

    async def _handle_aux_port(self, port: str, payload: Dict[str, Any]) -> None:
        # NB: `set_context` / `set_system_prompt` / `reset_history` /
        # `clear_context` are now async because the @rpc decorator on the
        # typed-RPC surface shadows the original sync setters. Calling
        # them without `await` here silently no-ops (the coroutine is
        # never iterated), which is the historical
        # `legacy aux-port raw-publish` bug. `await` makes the
        # raw-publish path equivalent to the typed-RPC path.
        if port == "context":
            text = payload.get("text")
            if isinstance(text, str):
                await self.set_context(text)
        elif port == "system_prompt":
            text = payload.get("text")
            if isinstance(text, str):
                await self.set_system_prompt(text)
        elif port == "reset":
            await self.reset_history()
        elif port == "barge_in":
            # `request_barge_in` is not shadowed (the @rpc method is
            # named `barge_in`, not `request_barge_in`), so the sync
            # call here still works.
            self.request_barge_in()
        else:
            logger.warning(
                "[%s] unknown aux port %r on LFM2AudioNode; payload ignored",
                self.node_id, port,
            )

    # ────── Main processing ───────────────────────────────────────────

    async def process(
        self, data: Any
    ) -> AsyncGenerator[Any, None]:
        """
        Route between aux-port control messages and audio user turns.

        Aux-port messages produce no outputs (they only update state).
        Audio user turns stream interleaved text + audio back.
        """
        # ── Unwrap the aux-port envelope (control-bus publishes) ──
        envelope = self._extract_envelope(data)
        if envelope is not None:
            port, payload = envelope
            logger.info(
                "[%s] aux-port envelope detected: port=%s payload=%r",
                self.node_id, port, payload,
            )
            await self._handle_aux_port(port, payload)
            return  # no yielded output for control-plane frames

        # ── Otherwise, treat the input as an audio user turn ──
        async for item in self._process_audio_turn(data):
            yield item

    async def _process_audio_turn(
        self, data: Any
    ) -> AsyncGenerator[Any, None]:
        if not _HAS_RUNTIME_DATA or RuntimeData is None:
            logger.error(
                "[%s] RuntimeData bindings unavailable in this worker — "
                "cannot emit text/audio output. Dropping turn.",
                self.node_id,
            )
            return

        # Dispatch on input modality. We accept either audio (the usual
        # ASR / interleaved-S2S input) or text (TTS mode, where the user
        # supplies a prompt and the model responds with synthesized audio).
        is_audio = hasattr(data, "is_audio") and data.is_audio()
        is_text = hasattr(data, "is_text") and data.is_text()

        if not is_audio and not is_text:
            kind = getattr(data, "data_type", lambda: type(data).__name__)()
            logger.error("Expected audio or text input, got %s", kind)
            yield RuntimeData.text(f"ERROR: expected audio or text input, got {kind}")
            return

        audio_array = None
        text_input: Optional[str] = None

        if is_audio:
            # Pull audio out of RuntimeData up front — the PyO3 handle is
            # not safe to touch after an async suspension point.
            if hasattr(data, "as_audio"):
                samples_bytes, input_sample_rate, _channels, _fmt, _n = data.as_audio()
                audio_array = np.frombuffer(samples_bytes, dtype=np.float32)
            else:
                payload = getattr(data, "payload", None)
                meta = getattr(data, "metadata", None)
                input_sample_rate = int(getattr(meta, "sample_rate", 0) or 0)

                if isinstance(payload, np.ndarray):
                    audio_array = payload.astype(np.float32, copy=False).reshape(-1)
                elif isinstance(payload, (bytes, bytearray, memoryview)):
                    audio_array = np.frombuffer(bytes(payload), dtype=np.float32)
                else:
                    yield RuntimeData.text(
                        f"ERROR: unsupported RuntimeData payload type "
                        f"{type(payload).__name__}"
                    )
                    return

            if input_sample_rate != self.sample_rate:
                yield RuntimeData.text(
                    f"ERROR: input sample rate {input_sample_rate}Hz "
                    f"does not match model rate {self.sample_rate}Hz"
                )
                return
        else:
            # Text input (TTS-style). Pull the prompt out before any await.
            try:
                text_input = data.as_text() if hasattr(data, "as_text") else None
            except Exception as e:
                logger.error("[%s] failed to decode text input: %s", self.node_id, e)
                text_input = None
            if not text_input:
                payload = getattr(data, "payload", None)
                if isinstance(payload, str):
                    text_input = payload
                elif isinstance(payload, (bytes, bytearray)):
                    try:
                        text_input = bytes(payload).decode("utf-8")
                    except UnicodeDecodeError:
                        text_input = None
            if not text_input:
                yield RuntimeData.text("ERROR: empty text input")
                return

        session_id = (
            data.session_id
            if hasattr(data, "session_id") and data.session_id
            else "default"
        )
        session_state = await self._get_or_create_session(session_id)
        chat = session_state.chat_state

        # Reset stale barge-in at the start of a new turn.
        self._interrupt = False

        # Add the user turn — audio or text depending on input modality.
        chat.new_turn("user")

        if is_audio:
            wav = torch.from_numpy(audio_array).float()
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)
            chat.add_audio(wav, self.sample_rate)
        else:
            chat.add_text(text_input)

        chat.end_turn()

        chat.new_turn("assistant")
        session_state.turn_count += 1

        logger.info(
            "[%s] starting generation for session=%s turn=%d",
            self.node_id,
            session_id,
            session_state.turn_count,
        )

        text_tokens_for_history: List[Any] = []
        audio_tokens_for_history: List[Any] = []
        modality_flags_for_history: List[Any] = []

        if self.text_only:
            token_generator = self._model.generate_sequential(
                **chat,
                max_new_tokens=self.max_new_tokens,
                text_temperature=None,
                text_top_k=None,
            )
        else:
            token_generator = self._model.generate_interleaved(
                **chat,
                max_new_tokens=self.max_new_tokens,
                audio_temperature=self.audio_temperature,
                audio_top_k=self.audio_top_k,
            )

        token_idx = 0

        # Streaming audio state.
        #
        # LFM2-Audio's final audio token is an end-of-audio marker. We cannot
        # safely decode the latest audio token until a later audio token proves
        # it was not final. So we keep one-token lookahead:
        #
        #   new audio token arrives
        #   previous pending token is now confirmed real audio
        #   current token becomes pending
        #
        # At generation end, pending_audio_token is dropped.
        pending_audio_token: Optional[Any] = None
        confirmed_audio_batch: List[Any] = []
        audio_batch_size = self.audio_batch_size
        # Optionally use a smaller threshold for *just the first* emitted
        # chunk this turn. None / >= audio_batch_size means "same as steady
        # state" (no early-emit special case).
        first_chunk_size = self.first_chunk_audio_batch_size
        first_chunk_active = (
            first_chunk_size is not None and first_chunk_size < audio_batch_size
        )

        emitted_audio_samples = 0

        def _normalize_audio_code_token(t: Any) -> torch.Tensor:
            t = t.detach()

            if t.dim() == 1:
                tok = t
            elif t.dim() == 2 and t.shape[0] == 1:
                tok = t.squeeze(0)
            elif t.dim() == 2 and t.shape[1] == 1:
                tok = t.squeeze(1)
            else:
                raise ValueError(f"Unexpected audio token shape: {tuple(t.shape)}")

            tok = tok.long()

            if tok.numel() != 8:
                raise ValueError(
                    f"Expected 8 Mimi codebooks, got "
                    f"shape={tuple(tok.shape)} numel={tok.numel()}"
                )

            # Validate on CPU before entering Mimi decode. Bad codes can trigger
            # CUDA device-side asserts, which poison the worker process.
            tok_cpu = tok.detach().cpu()
            min_code = int(tok_cpu.min().item())
            max_code = int(tok_cpu.max().item())

            if min_code < 0 or max_code >= 2048:
                raise ValueError(
                    f"Audio code out of Mimi range [0, 2047]: "
                    f"min={min_code} max={max_code} shape={tuple(tok.shape)}"
                )

            return tok

        def _decode_confirmed_audio_batch(batch: List[Any]) -> Optional[Any]:
            nonlocal emitted_audio_samples

            if not batch:
                return None

            try:
                tokens = [_normalize_audio_code_token(t) for t in batch]

                # Mimi expects [batch, codebooks, frames] == [1, 8, T].
                mimi_codes = torch.stack(tokens, dim=1).unsqueeze(0)

                mimi = self._processor.mimi
                try:
                    mimi_device = next(mimi.parameters()).device
                    mimi_codes = mimi_codes.to(mimi_device)
                except StopIteration:
                    pass

                with torch.no_grad():
                    waveform = mimi.decode(mimi_codes)[0]

                arr = waveform.detach().float().cpu().numpy()

                if arr.ndim == 2:
                    arr = arr[0]
                elif arr.ndim > 2:
                    arr = arr.reshape(-1)

                arr = np.ascontiguousarray(arr, dtype=np.float32)

                if numpy_to_audio is not None:
                    audio_rd = numpy_to_audio(arr, self.sample_rate, channels=1)
                else:
                    audio_rd = RuntimeData.audio(arr, self.sample_rate, channels=1)

                # Stamp content-time pts_us, matching the Kokoro streaming pattern.
                chunk_pts_us = (emitted_audio_samples * 1_000_000) // self.sample_rate
                emitted_audio_samples += int(arr.shape[0])

                md = getattr(audio_rd, "metadata", None)
                if md is not None:
                    existing = md.annotations or {}
                    existing.update({"pts_us": int(chunk_pts_us)})
                    md.annotations = existing

                return audio_rd

            except RuntimeError:
                # Do not swallow CUDA errors. A device-side assert usually poisons
                # the CUDA context, and continuing makes the next innocent CUDA call
                # crash mysteriously.
                logger.exception(
                    "Fatal CUDA error decoding confirmed LFM2 audio batch: "
                    "batch_shapes=%s",
                    [tuple(getattr(t, "shape", ())) for t in batch],
                )
                raise
            except Exception:
                logger.exception(
                    "Failed to decode confirmed LFM2 audio batch: batch_shapes=%s",
                    [tuple(getattr(t, "shape", ())) for t in batch],
                )
                raise

        while True:
            if self._interrupt:
                logger.info("[%s] barge-in latched — halting generation", self.node_id)
                self._interrupt = False
                break

            try:
                token = next(token_generator)
            except StopIteration:
                break

            token_idx += 1

            if token_idx % 10 == 0:
                await asyncio.sleep(0)

            if token.numel() == 1:
                text_tokens_for_history.append(token)
                modality_flags_for_history.append(LFMModality.TEXT)

                decoded = self._processor.text.decode(token)
                if decoded:
                    yield RuntimeData.text(decoded)
                    await asyncio.sleep(0)

            else:
                audio_tokens_for_history.append(token)
                modality_flags_for_history.append(LFMModality.AUDIO_OUT)

                # One-token lookahead. The previous pending token is now known
                # not to be the final end-of-audio marker.
                if pending_audio_token is not None:
                    confirmed_audio_batch.append(pending_audio_token)

                pending_audio_token = token

                # Use a smaller threshold for the first chunk only; revert
                # to `audio_batch_size` immediately after so steady-state
                # Mimi/IPC overhead is unaffected.
                emit_threshold = (
                    first_chunk_size if first_chunk_active else audio_batch_size
                )

                if len(confirmed_audio_batch) >= emit_threshold:
                    audio_rd = _decode_confirmed_audio_batch(confirmed_audio_batch)
                    confirmed_audio_batch.clear()

                    if audio_rd is not None:
                        yield audio_rd
                        await asyncio.sleep(0)
                        first_chunk_active = False

        # Decode any remaining confirmed tokens.
        #
        # Do NOT decode pending_audio_token. At generation end, it corresponds
        # to Liquid's final audio_out[-1] end-of-audio marker.
        if confirmed_audio_batch:
            audio_rd = _decode_confirmed_audio_batch(confirmed_audio_batch)
            confirmed_audio_batch.clear()

            if audio_rd is not None:
                yield audio_rd
                await asyncio.sleep(0)

        pending_audio_token = None

        # Terminal markers so the client can cut over between turns.
        yield RuntimeData.text("<|text_end|>")
        yield RuntimeData.text("<|audio_end|>")

        # Append this turn's tokens to chat history so the next turn has context.
        try:
            if text_tokens_for_history or audio_tokens_for_history:
                text_stack = (
                    torch.stack(text_tokens_for_history, 1)
                    if text_tokens_for_history
                    else torch.empty((1, 0), dtype=torch.long)
                )

                if self.text_only:
                    codebooks = (
                        getattr(self._processor, "codebooks", None)
                        or getattr(getattr(self._processor, "mimi", None), "codebooks", None)
                        or 8
                    )

                    audio_stack = torch.empty((codebooks, 0), dtype=torch.long)
                    history_modality_flags = modality_flags_for_history

                else:
                    # Drop final end-of-audio marker from chat history too.
                    audio_history_codes = (
                        audio_tokens_for_history[:-1]
                        if len(audio_tokens_for_history) > 1
                        else []
                    )

                    audio_stack = (
                        torch.stack(audio_history_codes, 1)
                        if audio_history_codes
                        else torch.empty((8, 0), dtype=torch.long)
                    )

                    # Remove the final AUDIO_OUT modality flag that corresponds
                    # to the dropped end-of-audio marker.
                    history_modality_flags = list(modality_flags_for_history)

                    if len(audio_tokens_for_history) > 1:
                        for i in range(len(history_modality_flags) - 1, -1, -1):
                            if history_modality_flags[i] == LFMModality.AUDIO_OUT:
                                del history_modality_flags[i]
                                break

                if history_modality_flags:
                    modality_values = [int(f.value) for f in history_modality_flags]
                    modality_tensor = torch.tensor(
                        modality_values,
                        dtype=torch.long,
                    ).unsqueeze(0)
                else:
                    modality_tensor = torch.empty((1, 0), dtype=torch.long)

                # Defensive sanity check before calling liquid_audio.ChatState.append().
                expected_modality_len = text_stack.shape[1] + audio_stack.shape[1]
                actual_modality_len = modality_tensor.shape[1]

                if actual_modality_len != expected_modality_len:
                    raise ValueError(
                        "LFM2 chat history shape mismatch before append: "
                        f"text={tuple(text_stack.shape)} "
                        f"audio={tuple(audio_stack.shape)} "
                        f"modality={tuple(modality_tensor.shape)} "
                        f"expected_modality_len={expected_modality_len} "
                        f"actual_modality_len={actual_modality_len}"
                    )

                chat.append(
                    text=text_stack,
                    audio_out=audio_stack,
                    modality_flag=modality_tensor,
                )

            chat.end_turn()

        except Exception as e:  # noqa: BLE001
            logger.error("Failed to append to chat history: %s", e, exc_info=True)
            try:
                chat.end_turn()
            except Exception:  # noqa: BLE001
                pass
        # ────── Introspection ─────────────────────────────────────────────

        def get_config(self) -> dict:
            return {
                "node_id": self.node_id,
                "node_type": "LFM2AudioNode",
                "hf_repo": self.hf_repo,
                "system_prompt": self._system_prompt,
                "device": self.device,
                "audio_temperature": self.audio_temperature,
                "audio_top_k": self.audio_top_k,
                "max_new_tokens": self.max_new_tokens,
                "sample_rate": self.sample_rate,
                "session_timeout_minutes": self.session_timeout_minutes,
                "text_only": self.text_only,
                "context_len": len(self._context),
                "active_sessions": len(self._sessions),
            }

        def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
            if session_id not in self._sessions:
                return None
            s = self._sessions[session_id]
            return {
                "session_id": s.session_id,
                "turn_count": s.turn_count,
                "created_at": s.created_at.isoformat(),
                "last_accessed": s.last_accessed.isoformat(),
            }

        def list_sessions(self) -> List[Dict[str, Any]]:
            return [self.get_session_info(sid) for sid in self._sessions.keys()]


# Full-path registry alias — mirrors the echo_python.py / moss_tts pattern
# so the FFI runner can resolve `LFM2AudioNode` by either bare name or
# `lfm2_audio_source.LFM2AudioNode`. Wrapped in try/except so the file is
# still importable if the multiprocessing package is missing (e.g. for
# documentation tooling).
try:
    from remotemedia.core.multiprocessing import _NODE_REGISTRY as _MP_REGISTRY
    _MP_REGISTRY[f"{LFM2AudioNode.__module__}.{LFM2AudioNode.__name__}"] = LFM2AudioNode
except ImportError:
    pass
