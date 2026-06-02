from io import BytesIO
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, util

_model = SentenceTransformer('all-MiniLM-L6-v2')


def extract_text_from_pdf_bytes(file_bytes: bytes) -> str:
    """LOADING phase. Input = raw bytes of an uploaded PDF. Output = its text."""
    reader = PdfReader(BytesIO(file_bytes))
    all_text = []
    for page in reader.pages:
        page_text = page.extract_text()
        all_text.append(page_text)
    return "\n".join(all_text)


def rank_resumes(job_description: str, resumes: list[dict]) -> list[dict]:
    """resumes = [{'name':..., 'text':...}] built from live uploads."""
    jd_emb = _model.encode(job_description)
    scored = []
    for r in resumes:
        r_emb = _model.encode(r["text"])
        score = util.cos_sim(jd_emb, r_emb).item()
        scored.append({"name": r["name"], "text": r["text"], "score": round(score, 3)})
    return sorted(scored, key=lambda x: x["score"], reverse=True)
