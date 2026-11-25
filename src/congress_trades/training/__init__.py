"""Distillation: a local teacher labels statements, small students are LoRA-tuned to match it,
then fused, converted to GGUF, and quantized for llama.cpp."""
