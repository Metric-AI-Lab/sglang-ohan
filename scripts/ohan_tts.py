#!/usr/bin/env python3
"""Prepare the Armenian OhanTTS checkpoint for the SGLang stream server.

The source checkpoint remains untouched. Only a small ``config.json`` is
generated; model weights are symlinked to avoid a second multi-gigabyte copy.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


BASE_MODEL = "k2-fsa/OmniVoice"
DEFAULT_CHECKPOINT = "Bagratuni/1ep-08-armenian-no-mask-ce-three"
EXPECTED_ARCH = "OhanTTSForConditionalGeneration"


def prepare(checkpoint: str, output_dir: Path, base_model: str = BASE_MODEL) -> Path:
    snapshot = Path(snapshot_download(checkpoint)).resolve()
    config_path = snapshot / "config.yaml"
    weights_path = snapshot / "model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(f"Expected config.yaml and model.safetensors in {snapshot}")

    import yaml

    checkpoint_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(checkpoint_config, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}")
    model_config = checkpoint_config.get("model_config")
    if model_config is None:
        model_sections = [
            section
            for section in checkpoint_config.values()
            if isinstance(section, dict) and "postprocess_output" in section
        ]
        if len(model_sections) != 1:
            raise ValueError("Expected exactly one model configuration with postprocessing settings")
        model_config = model_sections[0]
    if not isinstance(model_config, dict):
        raise ValueError("model_config must be a YAML mapping")

    base_snapshot = Path(snapshot_download(base_model)).resolve()
    base_config = json.loads((base_snapshot / "config.json").read_text(encoding="utf-8"))
    llm_config = base_config.get("llm_config")
    if not isinstance(llm_config, dict) or llm_config.get("model_type") != "qwen3":
        raise ValueError("Expected OhanTTS base config with a Qwen3 llm_config")

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared_config = {
        "architectures": [EXPECTED_ARCH],
        "model_type": "ohan_tts",
        "llm_config": llm_config,
        "num_audio_codebook": int(base_config.get("num_audio_codebook", 8)),
        "audio_vocab_size": int(base_config.get("audio_vocab_size", 1025)),
        "audio_mask_id": int(base_config.get("audio_mask_id", 1024)),
        "audio_codebook_weights": base_config.get("audio_codebook_weights"),
        "dtype": "bfloat16",
        "transformers_version": base_config.get("transformers_version"),
    }
    target_weights = output_dir / "model.safetensors"
    if target_weights.is_symlink() or target_weights.exists():
        if target_weights.resolve() != weights_path.resolve():
            raise FileExistsError(
                f"Refusing to replace existing checkpoint link: {target_weights}"
            )
    else:
        target_weights.symlink_to(weights_path)
    (output_dir / "config.json").write_text(
        json.dumps(prepared_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    metadata = {
        "checkpoint": checkpoint,
        "checkpoint_snapshot": str(snapshot),
        "base_model": base_model,
        "architecture": EXPECTED_ARCH,
        "postprocess_output": bool(model_config.get("postprocess_output", False)),
    }
    (output_dir / "integration_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "ohan_tts",
    )
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start the local SGLang server after preparing the checkpoint",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--model-name", help="API model ID (defaults to the Hugging Face checkpoint ID)"
    )
    args = parser.parse_args()
    prepared = prepare(args.checkpoint, args.output_dir, args.base_model)
    print(prepared, flush=True)
    if args.serve:
        binary = Path(sys.executable).parent / "sgl-omni"
        if not binary.is_file():
            raise FileNotFoundError(
                f"SGLang-Omni is missing from this Python environment: {binary}"
            )
        os.execv(
            binary,
            [
                str(binary),
                "serve",
                "--model-path",
                str(prepared),
                "--model-name",
                args.model_name or args.checkpoint,
                "--host",
                args.host,
                "--port",
                str(args.port),
                "--skip-server-warmup",
            ],
        )


if __name__ == "__main__":
    main()
