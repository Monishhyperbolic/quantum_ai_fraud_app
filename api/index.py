from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import PyPDF2
import os
import psycopg2 
from groq import Groq
from dotenv import load_dotenv
import uuid
import logging
import json
import re

# --- Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    logger.error("FATAL: GROQ_API_KEY not found in environment variables.")
client = Groq(api_key=GROQ_API_KEY)

# --- Vercel Postgres Database Connection ---
POSTGRES_URL = os.getenv("POSTGRES_URL")

def get_db_connection():
    """Establishes a connection to the Postgres database and ensures the table exists."""
    logger.info("Connecting to Postgres database...")
    try:
        conn = psycopg2.connect(POSTGRES_URL)
        with conn.cursor() as cur:
            cur.execute('''CREATE TABLE IF NOT EXISTS summaries
                         (id TEXT PRIMARY KEY, filename TEXT, summary TEXT, conclusion TEXT, project_ideas TEXT)''')
        conn.commit()
        logger.info("Database connection successful and table ensured.")
        return conn
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        raise HTTPException(status_code=500, detail="Could not connect to the database.")

# --- Core Helper Functions (Unchanged) ---
def extract_text_from_pdf(file_obj):
    logger.info("Extracting text from PDF...")
    try:
        reader = PyPDF2.PdfReader(file_obj)
        text = "".join(page.extract_text() or "" for page in reader.pages)
        if not text.strip():
            raise ValueError("No text could be extracted from the PDF.")
        logger.info(f"Extracted {len(text)} characters from PDF.")
        return text
    except Exception as e:
        logger.error(f"Error during PDF text extraction: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to extract text from PDF: {e}")

def generate_summary_and_conclusion(text: str):
    logger.info("Generating summary and conclusion...")
    prompt = f"Summarize the following research paper text (max 200 words) and provide a separate conclusion. Separate the summary and conclusion with the exact phrase 'Conclusion:'.\n\nText: {text[:4000]}"
    try:
        response = client.chat.completions.create(model="llama3-70b-8192", messages=[{"role": "user", "content": prompt}], max_tokens=500, temperature=0.7)
        output = response.choices[0].message.content
        parts = output.split("Conclusion:", 1)
        summary = parts[0].strip()
        conclusion = parts[1].strip() if len(parts) > 1 else "No specific conclusion was generated."
        return summary, conclusion
    except Exception as e:
        logger.error(f"Groq API error during summary generation: {e}")
        raise HTTPException(status_code=500, detail="Error communicating with AI for summary generation.")

def generate_project_ideas(summary: str):
    logger.info("Generating project ideas...")
    prompt = f"Based on the following research summary, list 3-5 innovative and feasible website project ideas. Each idea should be on a new line and described in a single, concise sentence.\n\nSummary:\n{summary}"
    try:
        response = client.chat.completions.create(model="llama3-70b-8192", messages=[{"role": "user", "content": prompt}], max_tokens=300, temperature=0.8)
        raw_ideas = response.choices[0].message.content.strip()
        return [idea.strip() for idea in raw_ideas.split("\n") if idea.strip()]
    except Exception as e:
        logger.error(f"Groq API error during project idea generation: {e}")
        raise HTTPException(status_code=500, detail="Error communicating with AI for project ideas.")

def clean_js_code(code: str):
    code = re.sub(r'const\s+([^=]+)\s*=\s*require\(\'([^\']+)\'\);?', r'// const \1 = require("\2"); // Commented out for browser compatibility', code)
    code = re.sub(r'import\s+[^;\n]+\s+from\s+[\'"](fs|path|http|module)[\'"];', '// Server-side import removed for browser', code)
    return code

def generate_website_code(project_idea: str):
    logger.info(f"Generating in-depth website code for: {project_idea}")
    prompt = f"""
Generate a complete, functional, and well-commented web application based on the following idea.
**Project Idea:** {project_idea}
**Requirements:**
1.  **Frontend (HTML/CSS/JS):** `index.html` (using Tailwind CSS), `styles.css`, `script.js` (using ES Modules and `fetch`).
2.  **Backend (FastAPI):** `app.py` with functional API endpoints.
3.  **Instructions:** A clear Markdown string explaining the project.
**Output Format:** A single, valid JSON object with keys: "frontend" (an object with "index_html", "styles_css", "script_js"), "backend", "instructions".
"""
    try:
        response = client.chat.completions.create(model="llama3-70b-8192", messages=[{"role": "user", "content": prompt}], max_tokens=4096, temperature=0.5, response_format={"type": "json_object"})
        content = response.choices[0].message.content
        data = json.loads(content)
        data["frontend"]["script_js"] = clean_js_code(data["frontend"]["script_js"])
        return data
    except Exception as e:
        logger.error(f"Error generating website code: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate website code from AI.")

def edit_website_code(original_code: dict, edit_request: str):
    logger.info(f"Editing website code with request: {edit_request}")
    prompt = f"""
Given this website's source code and an edit request, modify the code to implement the change. Return the complete, updated code in the same JSON format.
**Original Code:** {json.dumps(original_code)}
**Edit Request:** {edit_request}
"""
    try:
        response = client.chat.completions.create(model="llama3-70b-8192", messages=[{"role": "user", "content": prompt}], max_tokens=4096, temperature=0.5, response_format={"type": "json_object"})
        content = response.choices[0].message.content
        data = json.loads(content)
        data["frontend"]["script_js"] = clean_js_code(data["frontend"]["script_js"])
        return data
    except Exception as e:
        logger.error(f"Error editing website code: {e}")
        raise HTTPException(status_code=500, detail="Failed to edit website code with AI.")


# --- API Endpoints (Updated for Postgres) ---
@app.post("/api/summarize")
async def summarize_papers(files: list[UploadFile] = File(...)):
    results = []
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            for file in files:
                text = extract_text_from_pdf(file.file)
                summary, conclusion = generate_summary_and_conclusion(text)
                project_ideas = generate_project_ideas(summary)
                
                cur.execute(
                    "INSERT INTO summaries (id, filename, summary, conclusion, project_ideas) VALUES (%s, %s, %s, %s, %s)",
                    (str(uuid.uuid4()), file.filename, summary, conclusion, "|".join(project_ideas))
                )
                results.append({"filename": file.filename, "summary": summary, "conclusion": conclusion, "project_ideas": project_ideas})
        conn.commit()
    except Exception as e:
        logger.error(f"Error in /api/summarize endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")
    finally:
        if conn:
            conn.close()
    return JSONResponse(content=results)

@app.get("/api/summaries")
async def get_summaries():
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT id, filename, summary, conclusion, project_ideas FROM summaries ORDER BY filename")
            rows = cur.fetchall()
        summaries = [{"filename": r[1], "summary": r[2], "conclusion": r[3], "project_ideas": r[4].split("|") if r[4] else []} for r in rows]
    except Exception as e:
        logger.error(f"Error in /api/summaries endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")
    finally:
        if conn:
            conn.close()
    return JSONResponse(content=summaries)

@app.post("/api/generate-website")
async def generate_website_code_api(data: dict):
    project_idea = data.get("idea")
    if not project_idea:
        raise HTTPException(status_code=400, detail="Request body must include a project 'idea'.")
    result = generate_website_code(project_idea)
    result["project_idea"] = project_idea
    return JSONResponse(content=result)

@app.post("/api/edit-code")
async def edit_code_api(data: dict):
    original_code = data.get("original_code")
    edit_request = data.get("edit_request")
    if not original_code or not edit_request:
        raise HTTPException(status_code=400, detail="Request body must include 'original_code' and 'edit_request'.")
    result = edit_website_code(original_code, edit_request)
    return JSONResponse(content=result)