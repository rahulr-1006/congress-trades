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

## 1. Data

| Split | Source | Use |
| --- | --- | --- |
| `train` | Dev-period hearings (≤ 2023), statements that pass the mention pre-filter | LoRA training |
| `val` | Held-out hearings from the same period | Early stopping, prompt choices |
| `gold` | ~150 hand-labeled statements, never trained on | Extraction accuracy |
| `holdout` | Post-cutoff hearings (2024+) | Final trading evaluation only |

Splits are by **hearing**, not by statement, so the same hearing never leaks across splits.

Label sources (open decision):
- **Hand labels:** highest quality and slowest (~1 min/statement with a labeling CLI).
- **Deterministic silver labels for entities:** the mention pre-filter gives grounded tickers
  for free.
- **Teacher distillation:** a larger model labels the training set and the 8B student learns
  to imitate it. This is the standard trick, and the teacher is only used offline; inference
  stays fully local.

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
