# test_setup.py
print("Testing imports...")


import fitz; print("✅ PyMuPDF (fitz)")
import pdfplumber; print("✅ pdfplumber")
import torch; print(f"✅ PyTorch — CUDA: {torch.cuda.is_available()}")
from sentence_transformers import SentenceTransformer; print("✅ sentence-transformers")
import langchain; print("✅ LangChain")
import faiss; print("✅ FAISS")
import fastapi; print("✅ FastAPI")
from transformers import AutoTokenizer; print("✅ HuggingFace Transformers")

print("\n🎉 All good! Ready to build Vakeel AI.")