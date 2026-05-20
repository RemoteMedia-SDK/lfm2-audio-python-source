# lfm2-audio-python-source-plugin — LFM2.5-Audio as a Python source plugin

Two files do the real work:

```
lfm2-audio-python-source-plugin/
├── plugin.toml             ← metadata + per-plugin venv requires
└── lfm2_audio_source.py    ← LFM2AudioNode, vendored from the in-tree node
```

Plus the usual `manifest.json` / `consume.py` / `run.sh` for the smoke
demo.

This is the **Python source-load** sibling of
[`examples/lfm25-audio-loadable/`](../lfm25-audio-loadable/) (the
cdylib path). Same `LFM2AudioNode` semantics — interleaved
text + audio output, control-bus aux ports (`context`,
`system_prompt`, `reset`, `barge_in`), per-session conversation
history — distributed as plain Python instead of a compiled `.so`.

## How it loads

When a manifest references this directory (or a tagged GitHub release
of a repo with this layout), the resolver:

1. Parses `plugin.toml`.
2. Sees `language = "python"`.
3. Provisions a uv-managed venv from `[python].requires` (torch +
   CUDA + liquid-audio + transformers ≈ 5–7 GB on first run).
4. Spawns `python -m remotemedia.core.multiprocessing.runner` with
   `--module-root <this dir> --register-module lfm2_audio_source`.
5. Registers `LFM2AudioNode` into the executor's registry.

Same iceoryx2 IPC + READY handshake as the cdylib path. The plugin's
Python process is fully isolated from the host's in-tree
`remotemedia.nodes.ml.lfm2_audio` import — no conflict if both
register the same `node_type`.

## Try it

```bash
cd examples/lfm2-audio-python-source-plugin
./run.sh
```

First run is slow: uv has to download torch, liquid-audio,
transformers, and friends into the per-plugin venv. The script bumps
`REMOTEMEDIA_NODE_TIMEOUT_MS` to 300 s so the first session's cold
start (model download + Mimi codec init) doesn't trip the default
30 s guard. Subsequent runs reuse the venv and start in seconds.

The smoke `consume.py` exercises the **text-input TTS path** —
`LFM2AudioNode` accepts both audio (the usual ASR / interleaved S2S
input) and text (TTS mode where the user provides a prompt and the
model responds with synthesized speech). Three short prompts in, three
text + audio streams out. For an end-to-end audio→speech demo against
this same plugin, point one of the heavier examples at this manifest:

- [`examples/lfm2_audio_typed_rpc_chat.py`](../lfm2_audio_typed_rpc_chat.py)
  — typed-RPC control-bus chat with audio I/O.
- [`crates/transports/webrtc/examples/lfm2_audio_webrtc_server.rs`](../../crates/transports/webrtc/examples/lfm2_audio_webrtc_server.rs)
  — full mic→VAD→audio→speaker WebRTC pipeline.

## Publishing to GitHub

Same shape as `examples/python-source-plugin/`: tag a release, the
resolver fetches `plugin.toml` from
`https://raw.githubusercontent.com/{owner}/{repo}/{tag}/plugin.toml`
and the source tarball from
`https://codeload.github.com/{owner}/{repo}/tar.gz/refs/tags/{tag}`.
Consumers reference the plugin by spec:

```json
{
  "plugins": ["RemoteMedia-SDK/lfm2-audio-python-source@v0.1.0"]
}
```

No CI matrix needed (no per-platform binaries to build — Python source
is portable, the heavy native deps come from the platform's wheels at
venv provision time).

## Vendoring & upstream drift

`lfm2_audio_source.py` is a byte-for-byte vendor of
[`clients/python/remotemedia/nodes/ml/lfm2_audio.py`](../../clients/python/remotemedia/nodes/ml/lfm2_audio.py)
with a header note and a `_NODE_REGISTRY` full-path alias appended (so
the runner can resolve the class by either bare or
`lfm2_audio_source.LFM2AudioNode`). When the upstream changes:

```bash
cp clients/python/remotemedia/nodes/ml/lfm2_audio.py \
   examples/lfm2-audio-python-source-plugin/lfm2_audio_source.py
# then re-add the vendor header + registry-alias tail
```

External plugin authors maintaining a fork can drop the vendor note
and treat `lfm2_audio_source.py` as their own.

## Compare to `examples/lfm25-audio-loadable/`

| Aspect | `lfm25-audio-loadable/` (cdylib) | `lfm2-audio-python-source-plugin/` (source) |
|---|---|---|
| Languages plugin author writes | Rust + Python | **Python only** |
| Build step | `cargo build` per platform | None |
| Distribution | `.so` / `.dylib` / `.dll` (matrix per OS+arch) | git tag |
| Source visible to consumer | No (embedded in binary) | Yes (cloned) |
| Self-update on commit | Re-publish + re-resolve | Re-resolve same tag → re-fetch |
| First-load time | Fast (dlopen) — venv only for the embedded Python | Slower (tarball download + extract + venv provision) |
| Per-plugin venv | Yes (via uv) | Yes (via uv) |
| Aux-port + multi-output support | Yes | Yes (same IPC layer) |

Path 4 (cdylib + `python_plugin_export!`) stays useful for sealed
binary distribution. For Python-only plugins where source is fine to
be public, this source-load path is simpler.

## Cross-references

- [`examples/python-source-plugin/`](../python-source-plugin/) — minimal echo demo, same path.
- [`examples/lfm25-audio-loadable/`](../lfm25-audio-loadable/) — cdylib sibling.
- [`docs/CUSTOM_NODE_REGISTRATION.md`](../../docs/CUSTOM_NODE_REGISTRATION.md) — the four paths to ship a node.
- [`docs/LOADABLE_NODE_ABI.md`](../../docs/LOADABLE_NODE_ABI.md) — what crosses the dlopen boundary (relevant to the cdylib sibling; source-path bypasses it via IPC).
- [`crates/core/src/loadable/source_python.rs`](../../crates/core/src/loadable/source_python.rs) — `SourcePythonFactory` (resolver dispatch).
- [`crates/core/src/loadable/plugin_toml.rs`](../../crates/core/src/loadable/plugin_toml.rs) — `plugin.toml` schema.
