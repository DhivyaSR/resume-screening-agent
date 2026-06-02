import os, json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("GROQ_API_KEY"),
                base_url="https://api.groq.com/openai/v1", timeout=30)
MODEL = "llama-3.3-70b-versatile"


def extract_skills(resume_text):
    """STRUCTURED OUTPUT: skills as JSON. temperature 0 = consistent/parseable."""
    resp = client.chat.completions.create(
        model=MODEL, temperature=0,
        messages=[
            {"role":"system","content":
             "Extract key technical skills. Reply ONLY with a JSON array of strings."},
            {"role":"user","content": resume_text},
        ])
    raw = resp.choices[0].message.content.strip().replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except:
        return [raw]


def interview_questions(resume_text, job_description, n=4):
    """GENERATION: tailored questions. temperature 0.4 = natural variety."""
    resp = client.chat.completions.create(
        model=MODEL, temperature=0.4,
        messages=[
            {"role":"system","content":
             f"You are a technical interviewer. Write {n} tailored, numbered "
             "interview questions for this candidate and role."},
            {"role":"user","content": f"JOB:\n{job_description}\n\nRESUME:\n{resume_text}"},
        ])
    return resp.choices[0].message.content.strip()
