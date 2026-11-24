"""Local inference layer.

``engine`` knows how to run a model; ``tasks`` knows what to ask it. Prompts live as plain
files in ``prompts/`` so models and prompts can be swapped without touching pipeline code.
"""

from .engine import ChatModel, LlamaCppEngine
from .tasks import TASKS, TaskSpec, run_task

__all__ = ["ChatModel", "LlamaCppEngine", "TASKS", "TaskSpec", "run_task"]
