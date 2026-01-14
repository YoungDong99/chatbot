from dataclasses import dataclass
from typing import List

@dataclass
class Chunk:
    chunk_id: str
    text: str

def _split_by_separators(text: str, seps: List[str]) -> List[str]:
    """재귀 분할: 큰 구분자부터 쪼개고, 너무 크면 다음 구분자로 더 쪼갬"""
    if not seps:
        return [text]

    sep = seps[0]
    if sep == "":
        # 더 이상 쪼갤 구분자가 없으면 문자 단위로 강제 분할용 반환
        return list(text)

    parts = text.split(sep)
    out = []
    for p in parts:
        p = p.strip()
        if p:
            out.append(p)
    return out

def recursive_chunk_text(text: str, chunk_size: int, overlap: int) -> List[Chunk]:
    """
    Recursive 방식:
    - \n\n, \n, 문장끝(., ?, !), 공백 순으로 분할 시도
    - chunk_size를 넘으면 더 작은 구분자로 재귀적으로 분할
    - 최종적으로 chunk_size에 맞게 합치고 overlap 적용
    """
    text = text.replace("\r\n", "\n").strip()
    seps = ["\n\n", "\n", ". ", "? ", "! ", " "]

    # 1) 재귀적으로 쪼개서 '작은 조각' 리스트 만들기
    pieces = [text]
    for sep in seps:
        new_pieces = []
        for piece in pieces:
            if len(piece) <= chunk_size:
                new_pieces.append(piece)
            else:
                new_pieces.extend(_split_by_separators(piece, [sep]))
        pieces = new_pieces

    # 2) 작은 조각들을 chunk_size 이하로 "합치기"
    merged = []
    buf = ""
    for p in pieces:
        if not buf:
            buf = p
            continue
        # +1은 공백 하나 넣는 느낌
        if len(buf) + 1 + len(p) <= chunk_size:
            buf = buf + " " + p
        else:
            merged.append(buf.strip())
            buf = p
    if buf:
        merged.append(buf.strip())

    # 3) overlap 적용해서 최종 chunks 만들기
    chunks: List[Chunk] = []
    prev = ""
    for i, m in enumerate(merged):
        if prev:
            joined = (prev[-overlap:] + "\n" + m).strip()
        else:
            joined = m
        chunks.append(Chunk(chunk_id=f"c{i:04d}", text=joined))
        prev = joined

    return chunks
