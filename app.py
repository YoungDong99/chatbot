import streamlit as st
from pathlib import Path

from rag.config import (
    DATA_DOCX_PATH, FAISS_INDEX_PATH, CHUNKS_PATH, LLM_MODEL_PATH,
    EMB_MODEL_ID, EMB_MAX_LENGTH, TOP_K
)
from rag.build_index import ensure_llm_model, build as build_index
from rag.embeddings import Embedder
from rag.faiss_store import FaissStore
# from rag.llm_llamacpp import LlamaCppLLM
from rag.rag_chain import RAGChatbot

st.set_page_config(page_title="사내 챗봇 (RAG)", layout="wide")

@st.cache_resource
def load_embedder():
    return Embedder(model_id=EMB_MODEL_ID, max_length=EMB_MAX_LENGTH, device="cpu")

# @st.cache_resource
# def load_llm():
#     ensure_llm_model()
#     return LlamaCppLLM(model_path=LLM_MODEL_PATH, n_ctx=4096, max_tokens=512)

def load_store():
    if not (FAISS_INDEX_PATH.exists() and CHUNKS_PATH.exists()):
        return None
    return FaissStore.load(FAISS_INDEX_PATH, CHUNKS_PATH)

st.sidebar.title("설정")
docx_path_str = st.sidebar.text_input("테스트 DOCX 경로", value=str(DATA_DOCX_PATH))
docx_path = Path(docx_path_str)

if st.sidebar.button("인덱스 새로 만들기"):
    with st.spinner("인덱스 생성 중... (CPU는 느릴 수 있어요)"):
        build_index(docx_path)
    st.success("완료!")

store = load_store()
if store is None:
    st.warning("인덱스가 없습니다. 사이드바에서 '인덱스 새로 만들기'를 먼저 눌러주세요.")
    st.stop()

embedder = load_embedder()
# llm = load_llm()
llm = None # 임시 : LLM 없이 검색 결과만
bot = RAGChatbot(embedder=embedder, store=store, llm=llm, top_k=TOP_K)

st.title("업무 도우미 챗봇")                                                        

if "chat" not in st.session_state:
    st.session_state.chat = []

for role, content in st.session_state.chat:
    with st.chat_message(role):
        st.markdown(content)

q = st.chat_input("질문을 입력하세요")
if q:
    st.session_state.chat.append(("user", q))
    with st.chat_message("user"):
        st.markdown(q)

    with st.chat_message("assistant"):
        with st.spinner("검색 + 생성 중..."):
            result = bot.answer(q)
        st.markdown(result["answer"])

        with st.expander("근거(Top-K) 보기"):
            for s in result["sources"]:
                st.markdown(f"**[{s['chunk_id']}] score={s['score']:.3f}**")
                st.write(s["text"][:1200] + ("..." if len(s["text"]) > 1200 else ""))

    st.session_state.chat.append(("assistant", result["answer"]))
