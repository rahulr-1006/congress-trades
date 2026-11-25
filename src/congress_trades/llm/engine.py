"""llama.cpp inference engine (via ``llama-cpp-python``). Fully offline once a GGUF is on disk."""

from __future__ import annotations

from typing import Protocol


class ChatModel(Protocol):
    """Anything that turns (system, user) prompts into a completion string."""

    def chat(
        self, system: str, user: str, *, max_tokens: int = 1024, schema: dict | None = None
    ) -> str: ...


class LlamaCppEngine:
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        temperature: float = 0.1,
        seed: int = 42,
    ):
        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise RuntimeError('llama-cpp-python not installed: pip install -e ".[llm]"') from e
        # n_gpu_layers=-1 offloads every layer (Metal on Apple Silicon, CUDA on NVIDIA).
        self._llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            seed=seed,
            verbose=False,
        )
        self.temperature = temperature
        self.model_path = model_path
        self.last_usage: dict[str, int] = {}

    def chat(
        self, system: str, user: str, *, max_tokens: int = 1024, schema: dict | None = None
    ) -> str:
        # json_object alone guarantees syntactically valid JSON; adding a schema also
        # guarantees the keys and types (llama.cpp converts it to a GBNF grammar).
        fmt: dict = {"type": "json_object"}
        if schema:
            fmt["schema"] = schema
        resp = self._llm.create_chat_completion(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=self.temperature,
            max_tokens=max_tokens,
            response_format=fmt,
        )
        self.last_usage = resp.get("usage", {})
        return resp["choices"][0]["message"]["content"]
