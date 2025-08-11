from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import PyPDF2
import sqlite3
import os
from groq import Groq
from dotenv import load_dotenv
import uuid
import logging
import json
import time
import re

# --- Setup ---
# Basic logging configuration to see the server's activity.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the FastAPI application.
app = FastAPI()

# Configure CORS (Cross-Origin Resource Sharing) to allow the frontend
# (running on a different port) to communicate with this backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for development purposes.
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods.
    allow_headers=["*"],  # Allows all headers.
)

# Load environment variables from a .env file.
load_dotenv()
# Initialize the Groq client with the API key.
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    logger.error("FATAL: GROQ_API_KEY not found in .env file.")
    raise RuntimeError("GROQ_API_KEY not found in .env file. Please create a .env file and add your key.")
client = Groq(api_key=GROQ_API_KEY)


# --- Database Initialization ---
def init_db():
    """Initializes the SQLite database and creates the 'summaries' table if it doesn't exist."""
    logger.info("Initializing database...")
    conn = sqlite3.connect('summaries.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS summaries
                 (id TEXT PRIMARY KEY, filename TEXT, summary TEXT, conclusion TEXT, project_ideas TEXT)''')
    conn.commit()
    conn.close()
    logger.info("Database initialized successfully.")

init_db()


# --- Core Helper Functions ---
def extract_text_from_pdf(file_obj):
    """Extracts all text from an uploaded PDF file object."""
    logger.info("Extracting text from PDF...")
    try:
        reader = PyPDF2.PdfReader(file_obj)
        text = "".join(page.extract_text() or "" for page in reader.pages)
        if not text.strip():
            raise ValueError("No text could be extracted from the PDF. The file might be image-based or empty.")
        logger.info(f"Extracted {len(text)} characters from PDF.")
        return text
    except Exception as e:
        logger.error(f"Error during PDF text extraction: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to extract text from PDF: {e}")

def generate_summary_and_conclusion(text: str):
    """Sends text to Groq to generate a summary and conclusion."""
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
    """Sends a summary to Groq to generate website project ideas."""
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
    """Removes server-side require() statements from JS code to make it browser-compatible."""
    code = re.sub(r'const\s+([^=]+)\s*=\s*require\(\'([^\']+)\'\);?', r'// const \1 = require("\2"); // Commented out for browser compatibility', code)
    code = re.sub(r'import\s+[^;\n]+\s+from\s+[\'"](fs|path|http|module)[\'"];', '// Server-side import removed for browser', code)
    return code

def generate_website_code(project_idea: str):
    """Generates a full, in-depth codebase for a given project idea."""
    logger.info(f"Generating in-depth website code for: {project_idea}")
    prompt = f"""
Generate a complete, functional, and well-commented web application based on the following idea.

**Project Idea:** {project_idea}

**Requirements:**
1.  **Frontend (HTML/CSS/JS):**
    -   `index.html`: A well-structured and semantic HTML file using Tailwind CSS classes for styling.
    -   `styles.css`: A separate CSS file for any custom styles that go beyond Tailwind's utilities (e.g., complex animations).
    -   `script.js`: A single, self-contained JavaScript file using ES Modules. **Implement the actual application logic**, including DOM manipulation and making `fetch` calls to the backend API. Add comments explaining the code's functionality. The code should be immediately runnable, not just a placeholder.

2.  **Backend (FastAPI):**
    -   `app.py`: A functional FastAPI application.
    -   **Implement the actual backend logic for the API endpoints.** If the idea is a "Text Summarizer", the endpoint should actually perform summarization. Add placeholder logic only if the task is impossible (e.g., requires a real database).
    -   Include robust input validation with Pydantic and clear error handling.

3.  **Instructions:**
    -   A clear Markdown string explaining the project, its dependencies, and how to run it.

**Output Format:**
Return a single, valid JSON object with the following keys: "frontend" (an object with "index_html", "styles_css", "script_js"), "backend", "instructions".
"""
    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192", messages=[{"role": "user", "content": prompt}], max_tokens=4096, temperature=0.5, response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        data = json.loads(content)
        data["frontend"]["script_js"] = clean_js_code(data["frontend"]["script_js"])
        return data
    except Exception as e:
        logger.error(f"Error generating website code: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate website code from AI.")

def edit_website_code(original_code: dict, edit_request: str):
    """Modifies an existing codebase based on a user's request."""
    logger.info(f"Editing website code with request: {edit_request}")
    prompt = f"""
Given this website's source code and an edit request, modify the code to implement the change. Return the complete, updated code in the same JSON format. Ensure the logic remains functional and well-commented.

**Original Code:**
{json.dumps(original_code)}

**Edit Request:**
{edit_request}
"""
    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192", messages=[{"role": "user", "content": prompt}], max_tokens=4096, temperature=0.5, response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        data = json.loads(content)
        data["frontend"]["script_js"] = clean_js_code(data["frontend"]["script_js"])
        return data
    except Exception as e:
        logger.error(f"Error editing website code: {e}")
        raise HTTPException(status_code=500, detail="Failed to edit website code with AI.")


# --- API Endpoints ---
@app.post("/summarize")
async def summarize_papers(files: list[UploadFile] = File(...)):
    """Endpoint to upload PDFs, summarize them, and generate project ideas."""
    results = []
    for file in files:
        try:
            text = extract_text_from_pdf(file.file)
            summary, conclusion = generate_summary_and_conclusion(text)
            project_ideas = generate_project_ideas(summary)
            
            # Store results in the database
            conn = sqlite3.connect('summaries.db')
            c = conn.cursor()
            c.execute("INSERT INTO summaries VALUES (?, ?, ?, ?, ?)", (str(uuid.uuid4()), file.filename, summary, conclusion, "|".join(project_ideas)))
            conn.commit()
            conn.close()
            
            results.append({"filename": file.filename, "summary": summary, "conclusion": conclusion, "project_ideas": project_ideas})
        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {e}")
            # If one file fails, we raise an error for that file but could potentially continue with others.
            # For simplicity, we stop on the first error.
            raise HTTPException(status_code=500, detail=f"An error occurred while processing {file.filename}.")
    return JSONResponse(content=results)

@app.get("/summaries")
async def get_summaries():
    """Endpoint to retrieve all previously saved summaries from the database."""
    conn = sqlite3.connect('summaries.db')
    c = conn.cursor()
    c.execute("SELECT * FROM summaries ORDER BY filename")
    rows = c.fetchall()
    conn.close()
    summaries = [{"filename": r[1], "summary": r[2], "conclusion": r[3], "project_ideas": r[4].split("|") if r[4] else []} for r in rows]
    return JSONResponse(content=summaries)

@app.post("/generate-website")
async def generate_website_code_api(data: dict):
    """Endpoint to generate a full website codebase from a project idea."""
    project_idea = data.get("idea")
    if not project_idea:
        raise HTTPException(status_code=400, detail="Request body must include a project 'idea'.")
    
    result = generate_website_code(project_idea)
    # Include the original idea in the response for frontend context.
    result["project_idea"] = project_idea
    
    return JSONResponse(content=result)

@app.post("/edit-code")
async def edit_code_api(data: dict):
    """Endpoint to edit an existing codebase using a natural language request."""
    original_code = data.get("original_code")
    edit_request = data.get("edit_request")
    if not original_code or not edit_request:
        raise HTTPException(status_code=400, detail="Request body must include 'original_code' and 'edit_request'.")
    
    result = edit_website_code(original_code, edit_request)
    return JSONResponse(content=result)
