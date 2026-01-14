from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List
from .embeddings import Embedder
from .faiss_store import FaissStore
from .llm_llamacpp import LlamaCppLLM

def build_context(hits: List[tuple[float, Dict[str, Any]]]) -> str:
    parts = []
    for score, m in hits:
        parts.append(f"[{m['chunk_id']}] (score={score:.3f})\n{m['text']}")
    return "\n\n---\n\n".join(parts)

@dataclass
class RAGChatbot:
    embedder: Embedder
    store: FaissStore
    llm: LlamaCppLLM
    top_k: int = 4

    def answer(self, question: str) -> Dict[str, Any]:
        qvec = self.embedder.encode([question])[0]
        hits = self.store.search(qvec, self.top_k)
        context = build_context(hits)

        system = (
            "너는 문서를 기반으로 답변하는 업무 도움 챗봇이다.\n"
            "아래 [문서 발췌]에 근거가 있을 때만 답하고, 근거가 부족하면 "
            "'문서에서 근거를 찾을 수 없습니다.'라고 말한다.\n"
            "답변은 한국어로 간결하게 작성한다."
        )
        user = f"[문서 발췌]\n{context}\n\n[질문]\n{question}"

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        answer_text = self.llm.chat(messages)

        return {
            "answer": answer_text,
            "sources": [{"score": s, **m} for s, m in hits],
        }
