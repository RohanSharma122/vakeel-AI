# src/api/main.py
"""
Vakeel AI — FastAPI Backend
Exposes the RAG + LLM pipeline as a REST API.

Endpoints:
  POST /chat          — main consultation endpoint
  POST /upload-pdf    — upload and analyze a legal document
  GET  /health        — health check
  GET  /sources       — get available law sources
"""

import os
import sys
import json
import re
import fitz
import tempfile
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.rag.llm import VakeelAI

load_dotenv()

# ── Global instance (loaded once at startup) ─────────────
vakeel: VakeelAI = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load VakeelAI once when server starts."""
    global vakeel
    print("🚀 Starting Vakeel AI server...")
    vakeel = VakeelAI()
    print("✅ Server ready!")
    yield
    print("👋 Shutting down...")


# ── FastAPI app ───────────────────────────────────────────
app = FastAPI(
    title="Vakeel AI",
    description="AI-powered Indian Legal Consultant",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow React frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # in production, set to your frontend URL
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ─────────────────────────────

class ChatRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    session_id: Optional[str] = None  # for future chat history


class Source(BaseModel):
    score: float
    source: str
    act_name: str
    section_id: str
    court: str
    text_preview: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    query: str
    tokens_used: int


class PDFAnalysisResponse(BaseModel):
    filename: str
    doc_type: str
    summary: str
    key_sections: list[str]
    legal_advice: str
    extracted_text_preview: str


# ── Endpoints ─────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Check if the API is running."""
    return {
        "status":  "healthy",
        "model":   "llama-3.3-70b-versatile",
        "vectors": 200000,
        "version": "1.0.0",
    }


@app.get("/sources")
async def get_sources():
    """Return available law sources in the database."""
    return {
        "sources": [
            {"name": "Indian Penal Code, 1860",                          "sections": 1250},
            {"name": "Code of Criminal Procedure, 1973",                 "sections": 752},
            {"name": "Motor Vehicles Act, 1988",                         "sections": 328},
            {"name": "Hindu Marriage Act, 1955",                         "sections": 84},
            {"name": "Information Technology Act, 2000",                 "sections": 189},
            {"name": "Protection of Children from Sexual Offences, 2012","sections": 59},
            {"name": "Court Judgments (Supreme Court, High Courts)",     "sections": 70385},
            {"name": "Legal QA Dataset",                                 "sections": 608006},
        ],
        "total_vectors": 200000,
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main consultation endpoint.
    Takes a legal query and returns AI-generated legal advice.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    if len(request.query) > 2000:
        raise HTTPException(status_code=400, detail="Query too long (max 2000 chars)")

    try:
        result = vakeel.consult(
            query=request.query,
            top_k=request.top_k or 5,
        )

        # Format sources for response
        sources = []
        for s in result["sources"]:
            sources.append(Source(
                score=       round(s["score"], 4),
                source=      s.get("source", ""),
                act_name=    s.get("act_name", ""),
                section_id=  s.get("section_id", ""),
                court=       s.get("court", ""),
                text_preview=s.get("text", "")[:200],
            ))

        return ChatResponse(
            answer=      result["answer"],
            sources=     sources,
            query=       request.query,
            tokens_used= result["tokens"],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Consultation failed: {str(e)}")


@app.post("/upload-pdf", response_model=PDFAnalysisResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a legal PDF document for analysis.
    Extracts text, identifies document type, and provides legal advice.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        # Save uploaded file temporarily
        content = await file.read()
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        # Extract text from PDF
        doc  = fitz.open(tmp_path)
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        os.unlink(tmp_path)

        if len(text.strip()) < 100:
            raise HTTPException(
                status_code=400,
                detail="Could not extract text from PDF. It may be scanned."
            )

        # Clean text
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]{2,}', ' ', text)
        text = text[:8000]  # limit for LLM context

        # Detect document type
        doc_type = detect_document_type(text)

        # Generate legal analysis via Vakeel AI
        analysis_query = f"""
        I have uploaded a legal document. Please analyze it.

        Document type detected: {doc_type}
        Document content:
        {text[:3000]}

        Please:
        1. Summarize what this document is about
        2. Identify key legal sections or clauses
        3. Explain my rights and obligations under this document
        4. Highlight any concerning clauses I should be aware of
        5. Recommend what legal action I should take if needed
        """

        result = vakeel.consult(analysis_query, top_k=3)

        # Extract key sections mentioned
        sections = re.findall(r'Section\s+\d+[A-Za-z]?', result["answer"])
        sections = list(dict.fromkeys(sections))[:5]  # unique, max 5

        return PDFAnalysisResponse(
            filename=              file.filename,
            doc_type=              doc_type,
            summary=               result["answer"],
            key_sections=          sections,
            legal_advice=          result["answer"],
            extracted_text_preview=text[:500],
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF analysis failed: {str(e)}")


def detect_document_type(text: str) -> str:
    """Detect the type of legal document from its text."""
    text_lower = text.lower()

    doc_types = {
        "FIR (First Information Report)":      ["first information report", "fir no", "police station"],
        "Legal Notice":                         ["legal notice", "take notice", "failing which"],
        "Rental Agreement":                     ["rental agreement", "lease agreement", "landlord", "tenant"],
        "Employment Contract":                  ["employment", "terms of employment", "employee", "employer"],
        "Court Summons":                        ["summons", "you are hereby summoned", "appear before"],
        "Consumer Complaint":                   ["consumer complaint", "deficiency of service", "consumer forum"],
        "Sale Deed":                            ["sale deed", "vendor", "vendee", "convey and transfer"],
        "Cheque Bounce Notice":                 ["dishonour of cheque", "section 138", "negotiable instrument"],
        "Bail Application":                     ["bail application", "anticipatory bail", "regular bail"],
        "Affidavit":                            ["affidavit", "solemnly affirm", "deponent"],
        "Power of Attorney":                    ["power of attorney", "attorney", "authorise and appoint"],
        "Will / Testament":                     ["last will", "testament", "bequeath", "testator"],
    }

    for doc_type, keywords in doc_types.items():
        if any(kw in text_lower for kw in keywords):
            return doc_type

    return "Legal Document"


# ── Run server ────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,   # auto-restart on code changes
    )