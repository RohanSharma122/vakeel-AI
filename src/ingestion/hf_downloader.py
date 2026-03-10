# src/ingestion/hf_downloader.py
from datasets import load_dataset
from pathlib import Path

Path("data/raw/huggingface").mkdir(parents=True, exist_ok=True)

datasets_to_try = [
    "viber1/indian-law-dataset",
    "Exploration-Lab/ILC",
    "ninadn/indian-legal",
]

for ds_id in datasets_to_try:
    print(f"\n⬇️  Trying: {ds_id}")
    try:
        ds = load_dataset(ds_id, trust_remote_code=True)
        total = sum(len(ds[s]) for s in ds)
        save_path = "data/raw/huggingface/" + ds_id.replace("/", "_")
        ds.save_to_disk(save_path)
        print(f"✅ {ds_id}: {total} rows saved to {save_path}")
    except Exception as e:
        print(f"❌ {ds_id}: {e}")

print("\n✅ Done!")