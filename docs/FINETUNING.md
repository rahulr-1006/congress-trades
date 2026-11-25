# Fine-tuning plan

Goal: show the whole local-LLM stack end to end, not just API calls. That means training an
adapter, converting and quantizing the model ourselves, and measuring the accuracy / latency /
memory trade-offs with real numbers.

## Pipeline

```
labeled statements ──► LoRA fine-tune (MLX, Apple Silicon) ──► fuse adapter into base weights
       │                                                             │
       │                                     convert_hf_to_gguf.py (llama.cpp) ──► F16 GGUF
       │                                                             │
       │                                    llama-quantize ──► Q8_0 / Q5_K_M / Q4_K_M / Q3_K_M
       ▼                                                             │
  gold eval set ◄───────────── evaluate every (model × quant) on accuracy + latency + RAM
```

## Status

| Stage | Status | Command |
| --- | --- | --- |
| Teacher labeling (Qwen2.5-14B-Instruct Q4_K_M, local) | ✅ built, overnight run | `distill label` |
| Dataset build (hearing-level split, grounded-label cleaning) | ✅ | `distill build` |
| LoRA training (MLX) | ✅ smoke-tested | `distill train` |
| Fuse → GGUF (llama.cpp converter) → quantize | ✅ smoke-tested | `distill export` |
| Student-vs-teacher eval + latency | ✅ | `distill eval`, `latency` |
| Hand-labeled gold set | 📋 backlog | see below |

```bash
./scripts/setup_convert.sh                     # one time: llama.cpp converter env
congress-trades distill label --model models/qwen2.5-14b-instruct-q4_k_m.gguf
congress-trades distill build --teacher qwen2.5-14b-instruct-q4_k_m.gguf
congress-trades distill train --base mlx-community/Llama-3.2-1B-Instruct-bf16 --run runs/llama-1b
congress-trades distill export --base mlx-community/Llama-3.2-1B-Instruct-bf16 --run runs/llama-1b \
    --quants Q8_0 Q5_K_M Q4_K_M Q3_K_M
congress-trades distill eval --model runs/llama-1b/llama-1b-Q4_K_M.gguf \
    --teacher qwen2.5-14b-instruct-q4_k_m.gguf
```

### Problems found while building it

- **MLX's GGUF export can't be loaded by current llama.cpp.** `mlx_lm fuse --export-gguf`
  writes `tokenizer.ggml.token_type` with the wrong type and drops the chat template. The
  pipeline now fuses to HF safetensors and runs llama.cpp's own `convert_hf_to_gguf.py`.
- **transformers version mismatch.** MLX (transformers 5) writes
  `tokenizer_class: TokenizersBackend`; the converter pins transformers 4.57, which calls the
  same tokenizer `PreTrainedTokenizerFast`. The config is rewritten between steps.
- **Schema-constrained decoding.** A briefly trained 1B student often left out keys. With the
  task's JSON schema compiled to a llama.cpp grammar, plus `maxItems` caps on lists to stop
  runaway repetition, valid-output rate on the smoke set went from **0% → 62.5% → 100%**.
- **Teacher noise.** Even Llama 3.1 8B emitted tickers like `DOT` (Department of
  Transportation) and `""`. Grounding against the mention pre-filter dropped 30 ungrounded
  tickers from 30 smoke-test labels before training.

### Smoke-test numbers (M1 Pro, 16 GB)

| Model | p50 latency | decode tok/s | size |
| --- | --- | --- | --- |
| Llama 3.1 8B Q4_K_M (base) | 8.6 s | 11 | 4.9 GB |
| Llama 3.2 1B LoRA, Q4_K_M | 3.9 s | 25 | 0.81 GB |
| Llama 3.2 1B LoRA, Q8_0 | | | 1.32 GB |
| Llama 3.2 1B LoRA, F16 | | | 2.48 GB |

The 1B's 25 tok/s is far below what the hardware should do for a 1B model, so the next
systems task is profiling where the time goes (grammar sampling vs. prefill vs. decode).

## 1. Data

| Split | Source | Use |
| --- | --- | --- |
| `train` | Dev-period hearings (≤ 2023), statements that pass the mention pre-filter | LoRA training |
| `val` | Held-out hearings from the same period | Early stopping, prompt choices |
| `gold` | ~150 hand-labeled statements, never trained on | Extraction accuracy |
| `holdout` | Post-cutoff hearings (2024+) | Final trading evaluation only |

Splits are by **hearing**, not by statement, so the same hearing never leaks across splits.

Label source: **local teacher distillation.** Qwen2.5-14B-Instruct (Apache-2.0, Q4_K_M,
~9 GB) labels the training set on-device; 1B/3B Llama students learn to imitate it. No cloud
APIs anywhere, for labeling or inference.

### Backlog: hand-labeled gold set

Student-vs-teacher agreement measures how well a student copies the teacher, not whether
either is *right*. Planned:
- `congress-trades label` CLI: shows a statement with the teacher's prediction pre-filled;
  accept/edit with a few keystrokes (~30–60 s per statement).
- ~150 statements, stratified by committee and by teacher signal/no-signal.
- Report teacher **and** student accuracy against gold, plus inter-annotator agreement
  (relabel 30 after a week).

## 2. Training (MLX LoRA on a 16 GB M1 Pro)

- Base: Llama 3.1 8B Instruct at 4-bit (QLoRA) for memory; Llama 3.2 3B and 1B as faster students.
- LoRA rank 8–16 on attention projections, seq len ≤ 1024, batch 1 with gradient accumulation.
- Loss only on the assistant JSON (prompt tokens masked).
- Track: train/val loss, tokens/sec, peak memory.

## 3. Conversion and quantization

- Fuse adapter → HF safetensors → `convert_hf_to_gguf.py` → F16 GGUF.
- `llama-quantize` to Q8_0, Q5_K_M, Q4_K_M, Q3_K_M.
- Sanity check: perplexity on val statements per quant level.

## 4. Evaluation matrix

For each of {base, fine-tuned} × {1B, 3B, 8B} × {quant levels}:

| Metric | How |
| --- | --- |
| Entity precision / recall / F1 | vs. gold tickers |
| Stance accuracy, signal F1 | vs. gold labels |
| Valid-JSON rate | fraction of outputs that parse with all required keys |
| p50 / p95 latency, prefill & decode tok/s | `model_outputs` latency + token counts |
| Peak RSS | `/usr/bin/time -l` |
| Trading CAR / t-stat | `validate --scorer llm` on dev |

Deliverable: a Pareto chart of accuracy vs. latency, and the claim to test is
"a fine-tuned 3B at Q4 matches base 8B extraction quality at ~3x the speed."

## 5. Systems experiments (llama.cpp)

- **Prompt-prefix KV caching:** the system prompt is identical across calls, so reusing its
  KV cache skips most prefill work.
- **Speculative decoding:** a 1B draft model proposes tokens for the 8B to verify.
- **Context size vs. memory:** n_ctx effect on KV cache RAM and latency.
- **Batching:** throughput vs. latency when processing a whole hearing at once.
- **Grammar-constrained decoding:** GBNF JSON schema vs. free generation, comparing valid-JSON
  rate and speed.
