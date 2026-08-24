#!/usr/bin/env python3
"""Convert scb10x/typhoon-whisper-large-v3 (PyTorch safetensors) to MLX format.

Downloads the HF transformers-format model, remaps the weight keys to the
MLX whisper schema, and saves as ``weights.npz`` (the same format the
mlx-community/whisper-large-v3-mlx repo uses).

The converted model is saved to a local directory so it can be referenced
by WHISPER_MODEL without touching .env or any live config.

Usage:
    python3 tools/convert_typhoon_to_mlx.py [--repo <hf-repo>] [--out <dir>]

Defaults:
    repo  = typhoon-ai/typhoon-whisper-large-v3   (scb10x version is private)
    out   = /tmp/typhoon-whisper-large-v3-mlx/
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Key mapping: PyTorch (transformers) -> MLX (mlx-whisper)
# ---------------------------------------------------------------------------

_ENCODER_PREFIX = "model.encoder."
_DECODER_PREFIX = "model.decoder."
_LM_HEAD = "lm_head"

# Encoder block keys (PyTorch -> MLX sub-key)
_ENCODER_ATTN_MAP = {
    "q_proj": "attn.query",
    "k_proj": "attn.key",
    "v_proj": "attn.value",
    "out_proj": "attn.out",
}
_ENCODER_MLP_MAP = {
    "fc1": "mlp1",
    "fc2": "mlp2",
}
_ENCODER_NORM_MAP = {
    "self_attn_layer_norm": "attn_ln",
    "final_layer_norm": "mlp_ln",
}

# Decoder block keys
_DECODER_SELF_ATTN_MAP = {
    "q_proj": "attn.query",
    "k_proj": "attn.key",
    "v_proj": "attn.value",
    "out_proj": "attn.out",
}
_DECODER_CROSS_ATTN_MAP = {
    "q_proj": "cross_attn.query",
    "k_proj": "cross_attn.key",
    "v_proj": "cross_attn.value",
    "out_proj": "cross_attn.out",
}
_DECODER_MLP_MAP = {
    "fc1": "mlp1",
    "fc2": "mlp2",
}
_DECODER_NORM_MAP = {
    "self_attn_layer_norm": "attn_ln",
    "encoder_attn_layer_norm": "cross_attn_ln",
    "final_layer_norm": "mlp_ln",
}


def _strip_suffix(key: str) -> tuple[str, str]:
    """Split off .weight or .bias suffix, returning (base, suffix)."""
    if key.endswith(".weight"):
        return key[:-7], ".weight"
    if key.endswith(".bias"):
        return key[:-5], ".bias"
    return key, ""


def _remap_encoder_key(src: str) -> str | None:
    """Map a PyTorch encoder weight key to MLX format.

    Handles both ``.weight`` and ``.bias`` suffixes.
    """
    assert src.startswith(_ENCODER_PREFIX), f"unexpected encoder key: {src}"
    base, suffix = _strip_suffix(src[len(_ENCODER_PREFIX):])

    if base.startswith("layers."):
        # Block-level key
        parts = base.split(".", 2)  # ['layers', 'N', 'subkey']
        block_idx = parts[1]

        # self_attn.*
        if parts[2].startswith("self_attn."):
            sub = parts[2][len("self_attn."):]
            mlx_sub = _ENCODER_ATTN_MAP.get(sub)
            if mlx_sub:
                return f"encoder.blocks.{block_idx}.{mlx_sub}{suffix}"
            return None

        # MLP
        if parts[2] in _ENCODER_MLP_MAP:
            mlx_mlp = _ENCODER_MLP_MAP[parts[2]]
            return f"encoder.blocks.{block_idx}.{mlx_mlp}{suffix}"

        # Layer norm
        if parts[2] in _ENCODER_NORM_MAP:
            mlx_norm = _ENCODER_NORM_MAP[parts[2]]
            return f"encoder.blocks.{block_idx}.{mlx_norm}{suffix}"

        return None

    # Top-level encoder keys
    mapping = {
        "conv1": "encoder.conv1",
        "conv2": "encoder.conv2",
        # MLX whisper stores these WITHOUT .weight suffix in the npz
        # (tree_unflatten expects flat keys, not nested dicts)
        "embed_positions": "encoder._positional_embedding",
        "layer_norm": "encoder.ln_post",
    }
    mlx_path = mapping.get(base)
    if mlx_path:
        # Strip any .weight/.bias suffix for positional embedding key
        if mlx_path == "encoder._positional_embedding":
            return mlx_path
        return mlx_path + suffix
    return None


def _remap_decoder_key(src: str) -> str | None:
    """Map a PyTorch decoder weight key to MLX format.

    Handles both ``.weight`` and ``.bias`` suffixes.
    """
    assert src.startswith(_DECODER_PREFIX), f"unexpected decoder key: {src}"
    base, suffix = _strip_suffix(src[len(_DECODER_PREFIX):])

    if base.startswith("layers."):
        parts = base.split(".", 2)
        block_idx = parts[1]

        # self_attn.*
        if parts[2].startswith("self_attn."):
            sub = parts[2][len("self_attn."):]
            mlx_sub = _DECODER_SELF_ATTN_MAP.get(sub)
            if mlx_sub:
                return f"decoder.blocks.{block_idx}.{mlx_sub}{suffix}"
            return None

        # encoder_attn (cross-attention)
        if parts[2].startswith("encoder_attn."):
            sub = parts[2][len("encoder_attn."):]
            mlx_sub = _DECODER_CROSS_ATTN_MAP.get(sub)
            if mlx_sub:
                return f"decoder.blocks.{block_idx}.{mlx_sub}{suffix}"
            return None

        # MLP
        if parts[2] in _DECODER_MLP_MAP:
            mlx_mlp = _DECODER_MLP_MAP[parts[2]]
            return f"decoder.blocks.{block_idx}.{mlx_mlp}{suffix}"

        # Layer norm
        if parts[2] in _DECODER_NORM_MAP:
            mlx_norm = _DECODER_NORM_MAP[parts[2]]
            return f"decoder.blocks.{block_idx}.{mlx_norm}{suffix}"

        return None

    # Top-level decoder keys
    mapping = {
        "embed_tokens": "decoder.token_embedding",
        # MLX whisper stores positional embedding WITHOUT .weight suffix
        "embed_positions": "decoder.positional_embedding",
        "layer_norm": "decoder.ln",
    }
    mlx_path = mapping.get(base)
    if mlx_path:
        if mlx_path == "decoder.positional_embedding":
            return mlx_path
        return mlx_path + suffix
    return None


def _remap_lm_head(src: str) -> str | None:
    base, suffix = _strip_suffix(src)
    if base == _LM_HEAD:
        return f"decoder.token_embedding{suffix}"  # shared embedding
    return None


def convert_and_save(
    repo_id: str,
    out_dir: Path,
    *,
    dtype: str = "float16",
) -> None:
    """Download PyTorch safetensors, remap keys, save as MLX npz + config.

    ``dtype`` defaults to float16 — the official mlx-community whisper repos
    store fp16 weights, and mlx-whisper asserts audio/feature dtypes against
    the model's weight dtype (a float32 npz trips
    ``audio_features has an incorrect dtype`` once ``fp16=True`` is passed).
    """
    from huggingface_hub import hf_hub_download, list_repo_files
    from safetensors.torch import load_file

    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Download and parse the safetensors index ---
    index_path = hf_hub_download(repo_id, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)

    weight_map = index["weight_map"]
    # Group files
    shard_files = set()
    for k, fname in weight_map.items():
        shard_files.add(fname)

    log.info("Downloading %d safetensors shards ...", len(shard_files))
    shards: dict[str, dict[str, np.ndarray]] = {}
    for fname in shard_files:
        shard_path = hf_hub_download(repo_id, fname)
        shards[fname] = load_file(shard_path)
        log.info("  loaded %s (%d keys)", fname, len(shards[fname]))

    # --- 2. Remap all keys ---
    mlx_weights: dict[str, np.ndarray] = {}
    skipped: list[str] = []
    unmapped: list[str] = []

    for torch_key, fname in sorted(weight_map.items()):
        mlx_key: str | None

        if torch_key.startswith(_ENCODER_PREFIX):
            mlx_key = _remap_encoder_key(torch_key)
        elif torch_key.startswith(_DECODER_PREFIX):
            mlx_key = _remap_decoder_key(torch_key)
        elif torch_key.startswith(_LM_HEAD):
            mlx_key = _remap_lm_head(torch_key)
        else:
            skipped.append(torch_key)
            continue

        if mlx_key is None:
            continue  # key not in mapping (shouldn't happen for valid keys)

        weight = shards[fname][torch_key].numpy()
        # Conv kernels: PyTorch stores (out, in, k); MLX whisper weights use
        # (out, k, in). Verified against the official mlx-community npz.
        if mlx_key in ("encoder.conv1.weight", "encoder.conv2.weight"):
            weight = weight.transpose(0, 2, 1)
        if mlx_key in mlx_weights:
            log.warning("duplicate MLX key %r (from %s), overwriting", mlx_key, torch_key)
        mlx_weights[mlx_key] = weight

    # Track which PyTorch keys produced no MLX key (e.g., bias keys that
    # weren't present in the safetensors file).

    log.info("Mapped %d keys -> MLX format", len(mlx_weights))
    if skipped:
        log.info("Skipped %d keys (not encoder/decoder/lm_head): %s", len(skipped), skipped[:10])

    # --- 3. Write config.json ---
    config_path = hf_hub_download(repo_id, "config.json")
    with open(config_path) as f:
        torch_config = json.load(f)

    # Extract Whisper-specific dimensions from the transformers config
    # Whisper large-v3: d_model=1280, encoder_layers=32, decoder_layers=32,
    #   encoder_attention_heads=20, decoder_attention_heads=20,
    #   vocab_size=51866, num_mel_bins=128
    mlx_config = {
        "n_mels": torch_config.get("num_mel_bins", 128),
        "n_audio_ctx": torch_config.get("max_source_positions", 1500),
        "n_audio_state": torch_config.get("d_model", 1280),
        "n_audio_head": torch_config.get("encoder_attention_heads", 20),
        "n_audio_layer": torch_config.get("encoder_layers", 32),
        "n_vocab": torch_config.get("vocab_size", 51866),
        "n_text_ctx": torch_config.get("max_target_positions", 448),
        "n_text_state": torch_config.get("d_model", 1280),
        "n_text_head": torch_config.get("decoder_attention_heads", 20),
        "n_text_layer": torch_config.get("decoder_layers", 32),
        "model_type": "whisper",
    }

    config_out = out_dir / "config.json"
    with open(config_out, "w") as f:
        json.dump(mlx_config, f, indent=2)
    log.info("Wrote config.json -> %s", config_out)

    # --- 4. Copy tokenizer files ---
    for fname in ["tokenizer.json", "tokenizer_config.json", "vocab.json",
                   "merges.txt", "special_tokens_map.json", "added_tokens.json",
                   "normalizer.json", "preprocessor_config.json"]:
        try:
            src = hf_hub_download(repo_id, fname)
            shutil.copy(src, out_dir / fname)
        except Exception:
            log.warning("Could not copy %s (may not exist)", fname)

    # --- 5. Add MLX-specific alignment_heads ---
    # alignment_heads is not in the PyTorch source; copy from the reference
    # mlx-community model so the converted model loads identically.
    try:
        from huggingface_hub import hf_hub_download
        mlx_compat_path = hf_hub_download(
            "mlx-community/whisper-large-v3-mlx", "weights.npz"
        )
        compat = np.load(mlx_compat_path)
        if "alignment_heads" in compat:
            mlx_weights["alignment_heads"] = compat["alignment_heads"]
            log.info("Copied alignment_heads (%d keys)", len(mlx_weights))
    except Exception:
        log.warning("Could not copy alignment_heads from mlx-community; model may fail to load")

    # --- 6. Save as .npz ---
    npz_path = out_dir / "weights.npz"
    log.info("Saving %d weights to %s (%s) ...", len(mlx_weights), npz_path, dtype)
    target_dtype = np.dtype(dtype)
    mlx_weights = {
        key: w.astype(target_dtype) if w.dtype != target_dtype else w
        for key, w in mlx_weights.items()
    }
    np.savez(str(npz_path), **mlx_weights)
    log.info("Done. Model saved to %s (%d MB)", npz_path, npz_path.stat().st_size // (1024 * 1024))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Convert typhoon-whisper-large-v3 (PyTorch safetensors) to MLX format."
    )
    ap.add_argument(
        "--repo",
        default="typhoon-ai/typhoon-whisper-large-v3",
        help="HF repo ID (default: typhoon-ai/typhoon-whisper-large-v3; scb10x is private)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("/tmp/typhoon-whisper-large-v3-mlx"),
        help="Output directory (default: /tmp/typhoon-whisper-large-v3-mlx/)",
    )
    args = ap.parse_args()

    log.info("Converting %s -> %s", args.repo, args.out)
    convert_and_save(args.repo, args.out)
    log.info("Model ready. Set WHISPER_MODEL=%s to use it.", str(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
