from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DOCX_PATH = PROJECT_ROOT / "data" / "sample.docx"

INDEX_DIR = PROJECT_ROOT / "indexes"
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
CHUNKS_PATH = INDEX_DIR / "chunks.json"

MODEL_DIR = PROJECT_ROOT / "models"

# 임베딩 모델
EMB_MODEL_ID = "skt/A.X-Encoder-base"
EMB_MAX_LENGTH = 1024  # CPU 테스트용 보수값

# 청킹
CHUNK_SIZE_CHARS = 1000
CHUNK_OVERLAP_CHARS = 300

# 검색 top-k
TOP_K = 4

# (여기 파일은 models/ 아래에 직접 넣거나, build_index.py에서 자동 다운로드도 가능)
LLM_GGUF_REPO = "MLP-KTLim/llama-3-Korean-Bllossom-8B-gguf-Q4_K_M"
LLM_GGUF_FILENAME = "llama-3-Korean-Bllossom-8B-Q4_K_M.gguf"
LLM_MODEL_PATH = MODEL_DIR / LLM_GGUF_FILENAME
