from docx import Document
from pathlib import Path

def load_docx_text(path: Path) -> str:
    doc = Document(str(path))
    paras = []
    for p in doc.paragraphs:
        t = (p.text or "").strip()
        if t:
            paras.append(t)
    return "\n".join(paras)
