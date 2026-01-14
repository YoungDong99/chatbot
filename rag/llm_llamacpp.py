from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any
import os
from llama_cpp import Llama

@dataclass
class LlamaCppLLM:
    model_path: Path
    n_ctx: int = 4096
    n_threads: int = 0
    temperature: float = 0.2
    top_p: float = 0.9
    max_tokens: int = 512

    def __post_init__(self):
        threads = self.n_threads if self.n_threads > 0 else max(1, (os.cpu_count() or 8) - 1)
        self.llm = Llama(
            model_path=str(self.model_path),
            n_ctx=self.n_ctx,
            n_threads=threads,
            chat_format="llama-3",
            verbose=False,
        )

    def chat(self, messages: List[Dict[str, str]]) -> str:
        resp: Dict[str, Any] = self.llm.create_chat_completion(
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )
        return resp["choices"][0]["message"]["content"]
