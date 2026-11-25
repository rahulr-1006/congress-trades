"""Train → fuse → GGUF → quantize. MLX on Apple Silicon for training, llama.cpp for the rest."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

# llama.cpp checkout + a Python env with torch/transformers for its converter.
# `scripts/setup_convert.sh` creates both under tools/.
LLAMA_CPP_DIR = Path(os.environ.get("LLAMA_CPP_DIR", "tools/llama.cpp"))
CONVERT_PYTHON = Path(os.environ.get("CONVERT_PYTHON", "tools/convert-venv/bin/python"))

QUANT_TYPES = ("Q8_0", "Q5_K_M", "Q4_K_M", "Q3_K_M")


def _run(args: list[str]) -> None:
    print("$", " ".join(args), flush=True)
    subprocess.run(args, check=True)


def train(base: str, data_dir: Path, adapter_dir: Path, iters: int = 600, batch_size: int = 2,
          num_layers: int = 16, learning_rate: float = 1e-4, grad_checkpoint: bool = True) -> None:
    """LoRA fine-tune with loss on the assistant JSON only (``--mask-prompt``)."""
    args = [sys.executable, "-m", "mlx_lm", "lora", "--model", base, "--train",
            "--data", str(data_dir), "--adapter-path", str(adapter_dir),
            "--iters", str(iters), "--batch-size", str(batch_size),
            "--num-layers", str(num_layers), "--learning-rate", str(learning_rate),
            "--mask-prompt", "--steps-per-eval", "100", "--save-every", "100",
            "--max-seq-length", "2048"]
    if grad_checkpoint:
        args.append("--grad-checkpoint")
    _run(args)


def fuse_to_gguf(base: str, adapter_dir: Path, out_dir: Path) -> Path:
    """Merge LoRA weights into the base (MLX), then convert with llama.cpp's reference
    converter. (mlx_lm's own --export-gguf writes tokenizer metadata current llama.cpp
    rejects and drops the chat template, so we don't use it.)"""
    out_dir.mkdir(parents=True, exist_ok=True)
    fused, gguf = out_dir / "fused", out_dir / "model-f16.gguf"
    _run([sys.executable, "-m", "mlx_lm", "fuse", "--model", base,
          "--adapter-path", str(adapter_dir), "--save-path", str(fused), "--dequantize"])
    _fix_tokenizer_class(fused)
    if not CONVERT_PYTHON.exists():
        raise SystemExit("llama.cpp converter not set up. Run scripts/setup_convert.sh")
    _run([str(CONVERT_PYTHON), str(LLAMA_CPP_DIR / "convert_hf_to_gguf.py"), str(fused),
          "--outtype", "f16", "--outfile", str(gguf)])
    return gguf


def _fix_tokenizer_class(model_dir: Path) -> None:
    """transformers 5 writes ``TokenizersBackend``; llama.cpp's converter pins transformers 4,
    which calls the same fast tokenizer ``PreTrainedTokenizerFast``."""
    cfg_path = model_dir / "tokenizer_config.json"
    cfg = json.loads(cfg_path.read_text())
    if cfg.get("tokenizer_class") == "TokenizersBackend":
        cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
        cfg_path.write_text(json.dumps(cfg, indent=2))


def quantize(src: Path, dst: Path, qtype: str) -> Path:
    """Quantize a GGUF in-process via llama.cpp's C API (same code as `llama-quantize`)."""
    import llama_cpp

    params = llama_cpp.llama_model_quantize_default_params()
    params.ftype = getattr(llama_cpp, f"LLAMA_FTYPE_MOSTLY_{qtype}")
    rc = llama_cpp.llama_model_quantize(str(src).encode(), str(dst).encode(), params)
    if rc != 0:
        raise RuntimeError(f"llama_model_quantize failed ({rc}) for {qtype}")
    return dst
