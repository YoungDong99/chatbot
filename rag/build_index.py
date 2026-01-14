from __future__ import annotations
from pathlib import Path
from huggingface_hub import hf_hub_download

from .config import (
    DATA_DOCX_PATH, INDEX_DIR, FAISS_INDEX_PATH, CHUNKS_PATH,
    LLM_GGUF_REPO, LLM_GGUF_FILENAME, LLM_MODEL_PATH,
    EMB_MODEL_ID, EMB_MAX_LENGTH, CHUNK_SIZE_CHARS, CHUNK_OVERLAP_CHARS
)
from .docx_loader import load_docx_text
from .chunker import recursive_chunk_text
from .embeddings import Embedder
from .faiss_store import FaissStore

def ensure_llm_model():
    LLM_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if LLM_MODEL_PATH.exists():
        return

    # gguf 파일 자동 다운로드
    local_path = hf_hub_download(
        repo_id=LLM_GGUF_REPO,
        filename=LLM_GGUF_FILENAME,
        local_dir=str(LLM_MODEL_PATH.parent),
        local_dir_use_symlinks=False,
    )
    print(f"Downloaded GGUF to: {local_path}")

def build(docx_path: Path = DATA_DOCX_PATH):
    if not docx_path.exists():
        raise FileNotFoundError(f"docx not found: {docx_path}")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    text = load_docx_text(docx_path)
    chunks = recursive_chunk_text(text, CHUNK_SIZE_CHARS, CHUNK_OVERLAP_CHARS)
    meta = [{"chunk_id": c.chunk_id, "text": c.text} for c in chunks]

    embedder = Embedder(model_id=EMB_MODEL_ID, max_length=EMB_MAX_LENGTH, device="cpu")
    vectors = embedder.encode([c.text for c in chunks], batch_size=8)

    store = FaissStore.build(vectors, meta)
    store.save(FAISS_INDEX_PATH, CHUNKS_PATH)

    print(f"Index saved: {FAISS_INDEX_PATH}")
    print(f"Chunks saved: {CHUNKS_PATH}")
    print(f"Chunks: {len(chunks)}")

if __name__ == "__main__":
    ensure_llm_model()
    build()
