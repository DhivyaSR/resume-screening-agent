from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import screening, agent

BASE_DIR = Path(__file__).resolve().parent
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def home():
    return FileResponse(BASE_DIR / "index.html", media_type="text/html",
                        headers={"Cache-Control": "no-store"})

@app.get("/index.html")
def index():
    return FileResponse(BASE_DIR / "index.html", media_type="text/html",
                        headers={"Cache-Control": "no-store"})

@app.post("/screen")
async def screen(job_description: str = Form(...),
                 files: list[UploadFile] = File(...)):   # <- LIVE uploads
    # LOADING (dynamic): read each uploaded file's bytes -> text
    resumes = []
    for uf in files:
        raw = await uf.read()
        text = (screening.extract_text_from_pdf_bytes(raw)
                if uf.filename.lower().endswith(".pdf")
                else raw.decode("utf-8", errors="ignore"))
        if text.strip():
            resumes.append({"name": uf.filename.rsplit(".",1)[0], "text": text})
    if not resumes:
        return {"error": "No readable resumes uploaded."}

    # EMBEDDING + SIMILARITY + RANKING
    ranked = screening.rank_resumes(job_description, resumes)

    # GENERATION — only for the top 3 (keeps it fast & within rate limits)
    top = []
    for c in ranked[:3]:
        top.append({"name": c["name"], "score": c["score"],
                    "skills": agent.extract_skills(c["text"]),
                    "questions": agent.interview_questions(c["text"], job_description)})

    return {"ranking": [{"name": c["name"], "score": c["score"]} for c in ranked],
            "top": top}