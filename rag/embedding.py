from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

@dataclass
class Embedder:
    model_id: str
    max_length: int = 1024
    device: str = "cpu"

    def __post_init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        vecs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            outputs = self.model(**inputs)

            # Mean pooling (attention_mask 고려)
            last_hidden = outputs.last_hidden_state  # (B, T, H)
            attn = inputs["attention_mask"].unsqueeze(-1)  # (B, T, 1)
            summed = (last_hidden * attn).sum(dim=1)
            counts = attn.sum(dim=1).clamp(min=1)
            pooled = summed / counts

            # L2 normalize (내적(IP) 검색 == 코사인 유사도처럼 사용)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vecs.append(pooled.cpu().numpy().astype(np.float32))

        return np.vstack(vecs)
