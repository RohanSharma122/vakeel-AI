# src/rag/llm.py
"""
Vakeel AI — LLM Layer
Takes retrieved legal chunks + user query → generates lawyer-style response
using Groq API (LLaMA 3.1 70B).
"""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent.parent))
import os
import re
from groq import Groq
from dotenv import load_dotenv
from src.rag.retriever import LegalRetriever

load_dotenv()

# ── Config ───────────────────────────────────────────────
GROQ_MODEL   = "llama-3.3-70b-versatile"
MAX_TOKENS   = 1024
TEMPERATURE  = 0.3   # low = more factual, less creative (good for legal)

# ── System prompt — this defines Vakeel AI's personality ─
SYSTEM_PROMPT = """You are Vakeel AI, an expert Indian legal consultant with deep knowledge of Indian law. You have 20+ years of experience as a senior advocate.

Your expertise covers:
- Indian Penal Code (IPC) and Bharatiya Nyaya Sanhita (BNS) 2023
- Code of Criminal Procedure (CrPC) and BNSS 2023  
- Consumer Protection Act 2019
- RTI Act 2005, IT Act 2000, POCSO Act 2012
- Hindu Marriage Act, Muslim Personal Law, Special Marriage Act
- Transfer of Property Act, Registration Act
- Labour laws (Payment of Wages, Minimum Wages, Factories Act)
- Motor Vehicles Act, RERA 2016
- Constitutional law, Fundamental Rights

Your communication style:
- Speak like a senior Indian advocate — confident, clear, empathetic
- Always cite specific sections and acts (e.g., "Under Section 138 of the NI Act...")
- Reference relevant court judgments when applicable
- Give practical, actionable advice — not just theory
- Explain legal jargon in simple terms
- Warn about urgent deadlines (limitation periods, court timelines)
- Structure responses clearly: situation analysis → applicable law → recommended action
- End complex advice with: "This is AI-generated legal guidance. For filing cases, consult a registered advocate."

CRITICAL RULES:
1. Only use information from the provided legal context
2. Never fabricate section numbers or case names
3. If the context doesn't cover the query, say so honestly
4. Always mention the specific act and section number when citing law
5. If situation is urgent (domestic violence, arrest, etc.) — say so clearly"""


# ── Prompt builder ───────────────────────────────────────

def build_prompt(query: str, context: str) -> str:
    """
    Build the RAG prompt combining retrieved context + user query.
    The context comes from FAISS retrieval.
    """
    return f"""You have been provided with relevant sections from Indian law and court judgments to help answer this legal query.

LEGAL CONTEXT (retrieved from Indian law database):
{context}

USER'S LEGAL QUERY:
{query}

Based on the legal context above, provide expert legal guidance as Vakeel AI. 
- Cite specific sections from the context
- Give practical next steps
- Be empathetic but direct"""


# ── Response cleaner ─────────────────────────────────────

def clean_response(text: str) -> str:
    """Clean and format the LLM response."""
    # Remove any leftover HTML
    text = re.sub(r'<[^>]+>', '', text)
    # Clean excessive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ── Main VakeelAI class ──────────────────────────────────

class VakeelAI:
    """
    End-to-end Vakeel AI pipeline:
    User query → RAG retrieval → LLM generation → Legal response
    """

    def __init__(self):
        print("🚀 Initialising Vakeel AI...")
        
        # Load retriever
        self.retriever = LegalRetriever()
        
        # Load Groq client
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found in .env file!\n"
                "Get your free key at: https://console.groq.com"
            )
        self.client = Groq(api_key=api_key)
        print("✅ Groq client ready")
        print("✅ Vakeel AI is ready to consult!\n")

    def consult(self, query: str, top_k: int = 5, verbose: bool = False) -> dict:
        """
        Main consultation method.
        
        Args:
            query:   User's legal question
            top_k:   Number of context chunks to retrieve
            verbose: Print retrieved sources if True
            
        Returns:
            dict with 'answer', 'sources', 'query'
        """

        # ── Step 1: Retrieve relevant legal context
        results = self.retriever.search(query, top_k=top_k)

        if verbose:
            print(f"\n📚 Retrieved {len(results)} sources:")
            for i, r in enumerate(results, 1):
                print(f"  [{i}] {r['source']} | {r['act_name'] or r['court']} | score: {r['score']:.4f}")

        # ── Step 2: Format context for LLM
        context = self.retriever.format_context(results)

        # ── Step 3: Build prompt
        prompt = build_prompt(query, context)

        # ── Step 4: Generate response via Groq
        response = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )

        answer = clean_response(response.choices[0].message.content)

        return {
            "query":   query,
            "answer":  answer,
            "sources": results,
            "model":   GROQ_MODEL,
            "tokens":  response.usage.total_tokens,
        }

    def chat(self, query: str) -> str:
        """Simple one-liner — just returns the answer string."""
        result = self.consult(query, verbose=True)
        return result["answer"]


# ═══════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════

def test_vakeel_ai():
    """Test end-to-end pipeline with real legal queries."""

    vakeel = VakeelAI()

    test_cases = [
        "My employer has not paid my salary for 3 months. What legal action can I take?",
        "My landlord is refusing to return my security deposit of 50,000 rupees after I vacated the flat.",
        "I received a cheque of 2 lakhs from a customer but it bounced. What should I do?",
        "My neighbour is threatening me and my family. What legal protection do I have?",
    ]

    for query in test_cases:
        print(f"\n{'='*65}")
        print(f"👤 Client: {query}")
        print(f"{'='*65}")

        result = vakeel.consult(query, top_k=5, verbose=True)

        print(f"\n⚖️  Vakeel AI:\n")
        print(result["answer"])
        print(f"\n📊 Tokens used: {result['tokens']}")
        print(f"{'─'*65}")

        # Only test first query to save API calls
        break

    print("\n✅ Vakeel AI is working end-to-end!")
    print("▶️  Next step: build the FastAPI backend")


if __name__ == "__main__":
    test_vakeel_ai()