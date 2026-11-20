# SPDX-License-Identifier: Apache-2.0
"""Prepare a local SGLang model directory for OhanTTS-Streaming."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download

STREAMING_MODEL_ID = "Metric-AI/OhanTTS-Streaming-v0.1"
BASE_MODEL_ID = "Metric-AI/OhanTTS-v0.1"
STREAMING_ARCHITECTURE = "OhanTTSStreamingForConditionalGeneration"
DURATION_TOKEN_ID = 151676


def _link_files(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if path.name in {"config.json", "README.md", ".gitattributes"}:
            continue
        target = destination / path.name
        if target.exists() or target.is_symlink():
            target.unlink()
        os.symlink(path.resolve(), target, target_is_directory=path.is_dir())


def prepare(output: Path) -> Path:
    streaming_dir = Path(snapshot_download(STREAMING_MODEL_ID))
    base_dir = Path(snapshot_download(BASE_MODEL_ID, allow_patterns=["config.json"]))
    output.mkdir(parents=True, exist_ok=True)
    _link_files(streaming_dir, output)
    config = json.loads((base_dir / "config.json").read_text(encoding="utf-8"))
    config["architectures"] = [STREAMING_ARCHITECTURE]
    config["model_type"] = "ohan_tts_streaming"
    config["llm_config"]["vocab_size"] = DURATION_TOKEN_ID + 1
    config["duration_token_id"] = DURATION_TOKEN_ID
    config["num_duration_classes"] = 128
    (output / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(prepare(args.output))


if __name__ == "__main__":
    main()
