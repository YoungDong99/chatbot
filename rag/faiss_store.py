from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple
import json
import numpy as np
import faiss

@dataclass
class FaissStore:
    index: faiss.Index
    meta: List[Dict[str, Any]]

    @staticmethod
    def build(vectors: np.ndarray, meta: List[Dict[str, Any]]) -> "FaissStore":
        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)  # normalized vectors => cosine 유사도처럼 사용
        index.add(vectors)
        return FaissStore(index=index, meta=meta)

    def save(self, index_path: Path, meta_path: Path) -> None:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        meta_path.write_text(json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def load(index_path: Path, meta_path: Path) -> "FaissStore":
        index = faiss.read_index(str(index_path))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return FaissStore(index=index, meta=meta)

    def search(self, query_vec: np.ndarray, top_k: int) -> List[Tuple[float, Dict[str, Any]]]:
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        scores, idxs = self.index.search(query_vec.astype(np.float32), top_k)
        out = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            out.append((float(score), self.meta[int(idx)]))
        return out
