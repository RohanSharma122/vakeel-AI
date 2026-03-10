# src/ingestion/chunker.py
"""
Vakeel AI — Text Chunker
Converts parsed legal documents into RAG-ready chunks.

Why chunking matters:
- LLMs have context limits (~4000 tokens)
- Smaller chunks = more precise retrieval
- Overlap ensures no context is lost at boundaries

Chunk sizes for legal text:
- 512 tokens with 50 token overlap works well
- ~512 tokens ≈ ~400 words ≈ ~2000 characters
"""

import json
import re
from pathlib import Path
from tqdm import tqdm
from datasets import load_from_disk
import pandas as pd

PROCESSED_DIR  = Path("data/processed")
CHUNKS_DIR     = Path("data/chunks")
HF_DIR         = Path("data/raw/huggingface")
JUDGMENTS_DIR  = Path("data/raw/judgments")

CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

# Chunking config
CHUNK_SIZE    = 512   # characters (approx 128 tokens)
CHUNK_OVERLAP = 64    # overlap between chunks


# ═══════════════════════════════════════════════════════════
# CORE CHUNKER
# ═══════════════════════════════════════════════════════════

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list:
    """
    Split text into overlapping chunks.
    Tries to split at sentence boundaries for cleaner chunks.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start  = 0

    while start < len(text):
        end = start + chunk_size

        if end >= len(text):
            chunks.append(text[start:].strip())
            break

        # Try to split at sentence boundary (. or \n)
        split_pos = -1
        for sep in ['\n\n', '.\n', '. ', '\n']:
            pos = text.rfind(sep, start + chunk_size // 2, end)
            if pos != -1:
                split_pos = pos + len(sep)
                break

        if split_pos == -1:
            split_pos = end  # hard cut if no boundary found

        chunk = text[start:split_pos].strip()
        if len(chunk) > 50:  # skip tiny chunks
            chunks.append(chunk)

        start = split_pos - overlap  # move back by overlap for continuity

    return chunks


# ═══════════════════════════════════════════════════════════
# SOURCE 1: BARE ACTS
# ═══════════════════════════════════════════════════════════

def chunk_bare_acts() -> list:
    """Chunk all parsed bare act JSON files."""
    print("\n📜 Chunking Bare Acts...")

    all_chunks   = []
    json_files   = list(PROCESSED_DIR.glob("*.json"))

    if not json_files:
        print("   No parsed acts found. Run parser.py first.")
        return []

    for json_path in tqdm(json_files, desc="Acts"):
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        act_name = data["metadata"]["act_name"]
        sections = data["sections"]

        for section in sections:
            text      = section.get("text", "").strip()
            sec_id    = section.get("section_id", "unknown")
            title     = section.get("title", "")
            if len(text) < 50:
                continue

            # Add context prefix so the chunk is self-contained
            # e.g. "Indian Penal Code, 1860 | Section 302 | Punishment for murder"
            context_prefix = f"{act_name}"
            if title:
                context_prefix += f" | {title}"

            text_with_context = f"[{context_prefix}]\n{text}"
            chunks = chunk_text(text_with_context)

            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    "chunk_id":   f"act_{json_path.stem}_{sec_id}_chunk{i}",
                    "text":       chunk,
                    "source":     "bare_act",
                    "act_name":   act_name,
                    "section_id": sec_id,
                    "title":      title,
                    "doc_type":   "legislation",
                })

    print(f"   ✅ {len(all_chunks):,} chunks from bare acts")
    return all_chunks


# ═══════════════════════════════════════════════════════════
# SOURCE 2: COURT JUDGMENTS
# ═══════════════════════════════════════════════════════════

def chunk_judgments() -> list:
    """Chunk IndiaKanoon court judgment JSON files."""
    print("\n⚖️  Chunking Court Judgments...")

    all_chunks     = []
    judgment_files = list(JUDGMENTS_DIR.rglob("*.json"))

    if not judgment_files:
        print("   No judgment files found.")
        return []

    for jpath in tqdm(judgment_files, desc="Judgments"):
        try:
            with open(jpath, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        # Extract text — IndiaKanoon API returns 'doc' field
        text  = data.get("doc", "") or data.get("text", "")
        title = data.get("title", "Unknown Case")
        court = data.get("docsource", data.get("court", "Unknown Court"))
        topic = data.get("topic", "")
        doc_id = data.get("tid", jpath.stem)

        if len(text) < 100:
            continue

        # Clean judgment text
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]{2,}', ' ', text)

        # Add context prefix
        context_prefix = f"Court Judgment | {court} | {title}"
        text_with_context = f"[{context_prefix}]\n{text}"

        chunks = chunk_text(text_with_context)

        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "chunk_id": f"judgment_{doc_id}_chunk{i}",
                "text":     chunk,
                "source":   "court_judgment",
                "title":    title,
                "court":    court,
                "topic":    topic,
                "doc_type": "judgment",
            })

    print(f"   ✅ {len(all_chunks):,} chunks from judgments")
    return all_chunks


# ═══════════════════════════════════════════════════════════
# SOURCE 3: HUGGINGFACE DATASETS
# ═══════════════════════════════════════════════════════════

def chunk_hf_datasets() -> list:
    """Process HuggingFace Indian legal datasets."""
    print("\n📦 Processing HuggingFace Datasets...")

    all_chunks = []
    hf_dirs    = [d for d in HF_DIR.iterdir() if d.is_dir()] if HF_DIR.exists() else []

    if not hf_dirs:
        print("   No HuggingFace datasets found.")
        return []

    for ds_dir in hf_dirs:
        print(f"   Loading: {ds_dir.name}")
        try:
            ds = load_from_disk(str(ds_dir))

            # Handle DatasetDict (multiple splits)
            if hasattr(ds, 'keys'):
                splits = list(ds.keys())
                rows   = [ds[split] for split in splits]
                import datasets as hf_datasets
                combined = hf_datasets.concatenate_datasets(rows)
            else:
                combined = ds

            # Get column names to find text field
            cols    = combined.column_names
            print(f"   Columns: {cols}")

            # Find the text column (case-insensitive search)
            cols_lower = {c.lower(): c for c in cols}
            text_col   = None
            ans_col    = None

            for candidate in ['text', 'instruction', 'content', 'judgment', 'document', 'input', 'question', 'context']:
                if candidate in cols_lower:
                    text_col = cols_lower[candidate]
                    break

            for candidate in ['response', 'output', 'answer', 'summary', 'label']:
                if candidate in cols_lower:
                    ans_col = cols_lower[candidate]
                    break

            if not text_col:
                print(f"   No text column found in {cols}, skipping")
                continue

            print(f"   Using text='{text_col}' answer='{ans_col}' ({len(combined):,} rows)")

            for i, row in enumerate(tqdm(combined, desc=f"  {ds_dir.name}", leave=False)):
                text = str(row.get(text_col, "")).strip()
                if len(text) < 100:
                    continue

                # Add any available label/answer as context
                answer = str(row.get(ans_col, "")).strip() if ans_col else ""

                # If it's a QA pair, format it as context
                if answer and len(str(answer)) > 10:
                    text = f"Question: {text}\nAnswer: {answer}"

                chunks = chunk_text(text)
                for j, chunk in enumerate(chunks):
                    all_chunks.append({
                        "chunk_id": f"hf_{ds_dir.name}_{i}_chunk{j}",
                        "text":     chunk,
                        "source":   "huggingface",
                        "dataset":  ds_dir.name,
                        "doc_type": "legal_qa",
                    })

            print(f"   ✅ {ds_dir.name}: processed")

        except Exception as e:
            print(f"   ❌ Failed {ds_dir.name}: {e}")

    print(f"   ✅ {len(all_chunks):,} chunks from HuggingFace")
    return all_chunks


# ═══════════════════════════════════════════════════════════
# SAVE ALL CHUNKS
# ═══════════════════════════════════════════════════════════

def save_chunks(all_chunks: list):
    """Save all chunks to JSONL file (one chunk per line — efficient for large datasets)."""

    output_path = CHUNKS_DIR / "all_chunks.jsonl"

    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"\n✅ Saved {len(all_chunks):,} chunks → {output_path}")

    # Also save a small sample for inspection
    sample_path = CHUNKS_DIR / "sample_chunks.json"
    import random
    sample = random.sample(all_chunks, min(20, len(all_chunks)))
    with open(sample_path, "w", encoding="utf-8") as f:
        json.dump(sample, f, ensure_ascii=False, indent=2)
    print(f"✅ Sample saved → {sample_path}")

    return output_path


# ═══════════════════════════════════════════════════════════
# STATS
# ═══════════════════════════════════════════════════════════

def print_stats(all_chunks: list):
    print(f"\n{'='*60}")
    print("CHUNKING COMPLETE — DATASET STATS")
    print(f"{'='*60}")

    # Count by source
    from collections import Counter
    source_counts = Counter(c["source"] for c in all_chunks)

    print(f"\n{'Source':<25} {'Chunks':>10}")
    print("-" * 37)
    for source, count in source_counts.most_common():
        bar = "█" * (count // 500)
        print(f"{source:<25} {count:>10,}  {bar}")

    # Avg chunk length
    avg_len = sum(len(c["text"]) for c in all_chunks) / len(all_chunks)
    total_tokens = sum(len(c["text"].split()) for c in all_chunks)

    print(f"\n{'='*60}")
    print(f"Total chunks:        {len(all_chunks):,}")
    print(f"Avg chunk length:    {avg_len:.0f} chars")
    print(f"Est. total tokens:   {total_tokens:,}")
    print(f"{'='*60}")
    print(f"\n Next step: run src/rag/embedder.py to build the vector index!")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*60)
    print("VAKEEL AI — CHUNKER")
    print("="*60)

    all_chunks = []

    # Chunk all three sources
    all_chunks += chunk_bare_acts()
    all_chunks += chunk_judgments()
    all_chunks += chunk_hf_datasets()

    if not all_chunks:
        print("\nNo chunks created. Check your data directories.")
    else:
        save_chunks(all_chunks)
        print_stats(all_chunks)