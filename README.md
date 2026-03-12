# ⚖️ Vakeel AI — Indian Legal Consultant

> An end-to-end RAG-powered AI system that provides expert legal guidance on Indian law using real court judgments, bare acts, and a 684,000+ chunk knowledge base.

![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi)
![LLaMA](https://img.shields.io/badge/LLaMA-3.3--70B-orange?style=flat-square)
![FAISS](https://img.shields.io/badge/FAISS-Vector--Store-red?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 🎯 What It Does

Vakeel AI answers complex Indian legal questions like a senior advocate — citing specific sections, referencing real court judgments, and giving actionable next steps.

**Example:**
> *"My employer hasn't paid my salary for 3 months. What can I do?"*

**Vakeel AI responds:**
> *"Under Section 33(2) of the Industrial Disputes Act and the Payment of Wages Act, your employer is in violation... You can: (1) Send a legal notice, (2) File a complaint with the Labour Commissioner, (3) Approach the Labour Court. Interest is automatic after 1 month delay per the Supreme Court ruling in..."*

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────┐
│           FastAPI Backend                   │
│                                             │
│  ┌─────────────┐    ┌──────────────────┐   │
│  │  e5-base-v2 │    │   FAISS Index    │   │
│  │  Embeddings │───▶│  200K vectors    │   │
│  └─────────────┘    └────────┬─────────┘   │
│                              │ Top-5 chunks │
│                    ┌─────────▼─────────┐   │
│                    │  LLaMA 3.3 70B    │   │
│                    │  (Groq API)       │   │
│                    └─────────┬─────────┘   │
└──────────────────────────────┼─────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Legal Response     │
                    │  + Source Citations │
                    └─────────────────────┘
```

---

## 📊 Dataset

| Source | Volume | Description |
|--------|--------|-------------|
| Bare Acts (PDFs) | 2,662 sections | IPC, CrPC, IT Act, Hindu Marriage Act, Motor Vehicles Act, POCSO |
| Court Judgments | 70,385 chunks | Supreme Court, High Courts via IndiaKanoon API |
| Legal QA Dataset | 608,006 chunks | HuggingFace Indian legal datasets |
| **Total** | **684,343 chunks** | **200,000 embedded vectors (e5-base-v2)** |

---

## 🚀 Features

- **⚖️ Legal Consultation** — Ask any Indian legal question in plain English
- **📚 Source Attribution** — Every answer cites the exact section and court
- **📄 PDF Analysis** — Upload FIR, legal notice, rental agreement, etc. for instant analysis
- **🔍 Semantic Search** — e5-base-v2 embeddings with FAISS for precise retrieval
- **🏛️ Multi-source RAG** — Searches across bare acts + judgments + legal QA simultaneously
- **⚡ Fast** — Sub-3 second responses via Groq API

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Embeddings | `intfloat/e5-base-v2` (768-dim) |
| Vector Store | FAISS IndexFlatIP |
| LLM | LLaMA 3.3 70B via Groq API |
| Backend | FastAPI + Uvicorn |
| PDF Parsing | PyMuPDF (fitz) |
| Frontend | Vanilla HTML/CSS/JS |
| Data | IndiaKanoon API + HuggingFace |

---

## 📁 Project Structure

```
vakeel-ai/
├── src/
│   ├── ingestion/
│   │   ├── parser.py          # PDF bare acts parser (4-strategy section splitter)
│   │   ├── chunker.py         # Text chunker with overlap (512 chars, 64 overlap)
│   │   └── hf_downloader.py   # HuggingFace dataset downloader
│   ├── rag/
│   │   ├── retriever.py       # FAISS retriever with filtering
│   │   ├── llm.py             # LLM layer + prompt builder + VakeelAI class
│   │   └── clean_metadata.py  # HTML cleaner for judgment text
│   └── api/
│       └── main.py            # FastAPI backend (4 endpoints)
├── vakeel-frontend/
│   └── index.html             # Chat UI
├── data/
│   ├── raw/                   # PDFs, judgments, HF datasets (not in git)
│   ├── processed/             # Parsed sections JSON (not in git)
│   ├── chunks/                # 684K chunks JSONL (not in git)
│   └── vector_store/          # FAISS index + metadata (not in git)
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/RohanSharma122/vakeel-ai.git
cd vakeel-ai
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Set up environment
```bash
# Create .env file
GROQ_API_KEY=your_groq_api_key       # https://console.groq.com
INDIANKANOON_TOKEN=your_ik_token     # https://api.indiankanoon.org
HF_TOKEN=your_hf_token               # https://huggingface.co/settings/tokens
```

### 3. Build the data pipeline
```bash
# Parse bare act PDFs
python src/ingestion/parser.py

# Build chunks
python src/ingestion/chunker.py

# Download HuggingFace datasets
python src/ingestion/hf_downloader.py
```

### 4. Build vector store (Google Colab recommended)
```python
# Upload data/chunks/all_chunks.jsonl to Google Drive
# Run notebooks/embedder.ipynb on Colab T4 GPU (~25 mins)
# Download vector_store/ back to data/vector_store/
```

### 5. Start the server
```bash
$env:PYTHONPATH = "C:\vakeel-ai"   # Windows PowerShell
python src/api/main.py
```

### 6. Open the UI
Open `vakeel-frontend/index.html` in your browser.

---

## 🔌 API Reference

### `POST /chat`
```json
// Request
{
  "query": "My landlord is not returning my security deposit",
  "top_k": 5
}

// Response
{
  "answer": "Under the relevant Rent Control Act...",
  "sources": [
    {
      "score": 0.8767,
      "source": "court_judgment",
      "act_name": "",
      "court": "State of Rajasthan - Act",
      "text_preview": "security deposit is not refunded..."
    }
  ],
  "tokens_used": 1243
}
```

### `POST /upload-pdf`
Upload a legal PDF (FIR, notice, agreement) for instant analysis.

### `GET /health`
Returns server status and vector count.

### `GET /sources`
Returns all available law databases.

Interactive docs: `http://localhost:8000/docs`

---

## 📈 Performance

| Metric | Value |
|--------|-------|
| Avg retrieval score | 0.85 - 0.89 |
| Avg response time | < 3 seconds |
| Vector dimensions | 768 |
| Index type | FAISS FlatIP (exact search) |
| Total legal knowledge | 684,343 chunks |

---

## 🗺️ Roadmap

- [x] PDF parsing pipeline
- [x] Multi-source chunking (684K chunks)
- [x] e5-base-v2 embeddings + FAISS index
- [x] RAG retrieval with source attribution
- [x] LLM integration (LLaMA 3.3 70B)
- [x] FastAPI backend
- [x] Chat UI with PDF upload
- [ ] Fine-tuned LLaMA on Indian legal QA (QLoRA)
- [ ] Hindi language support
- [ ] HuggingFace Spaces deployment
- [ ] Conversation memory (multi-turn chat)
- [ ] Lawyer directory integration

---

## ⚠️ Disclaimer

Vakeel AI provides AI-generated legal information for educational purposes only. It is not a substitute for advice from a qualified legal professional. For filing cases or official legal matters, consult a registered advocate.

---

## 👨‍💻 Author

**Aditya Mani**
- GitHub: [@RohanSharma122](https://github.com/RohanSharma122)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.