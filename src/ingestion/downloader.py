# src/ingestion/downloader.py
import requests
import os
from pathlib import Path

# Using direct/alternate sources that don't redirect
INDIAN_LAWS = {
    "ipc_1860": "https://legislative.gov.in/sites/default/files/A1860-45.pdf",
    "crpc_1973": "https://legislative.gov.in/sites/default/files/A1973-2.pdf",
    "consumer_protection_2019": "https://legislative.gov.in/sites/default/files/A2019-35.pdf",
    "rti_act_2005": "https://legislative.gov.in/sites/default/files/A2005-22.pdf",
    "domestic_violence_2005": "https://legislative.gov.in/sites/default/files/A2005-43.pdf",
    "it_act_2000": "https://legislative.gov.in/sites/default/files/A2000-21.pdf",
    "hindu_marriage_act_1955": "https://legislative.gov.in/sites/default/files/A1955-25.pdf",
    "pocso_act_2012": "https://legislative.gov.in/sites/default/files/A2012-32.pdf",
    "negotiable_instruments_act": "https://legislative.gov.in/sites/default/files/A1881-26.pdf",
    "transfer_of_property_act": "https://legislative.gov.in/sites/default/files/A1882-04.pdf",
}

def download_laws(save_dir="data/raw"):
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/pdf,*/*",
        "Referer": "https://legislative.gov.in/",
    })

    success = 0
    failed = []

    for name, url in INDIAN_LAWS.items():
        save_path = Path(save_dir) / f"{name}.pdf"

        if save_path.exists() and save_path.stat().st_size > 1000:
            print(f"⏭️  Already exists: {name}")
            success += 1
            continue

        print(f"⬇️  Downloading {name}...")
        try:
            response = session.get(url, timeout=30, allow_redirects=True)

            if response.status_code == 200 and len(response.content) > 1000:
                with open(save_path, "wb") as f:
                    f.write(response.content)
                size_kb = len(response.content) // 1024
                print(f"✅ {name} ({size_kb} KB)")
                success += 1
            else:
                print(f"❌ Failed {name} — HTTP {response.status_code}")
                failed.append(name)

        except Exception as e:
            print(f"❌ Error on {name}: {e}")
            failed.append(name)

    print(f"\n{'='*50}")
    print(f"✅ Downloaded: {success}/{len(INDIAN_LAWS)}")

    if failed:
        print(f"\n⚠️  Failed ({len(failed)}):")
        for f in failed:
            print(f"   - {f}")
        print("\n💡 For failed ones, download manually:")
        print("   1. Go to https://legislative.gov.in/acts-of-parliament-from-1836-to-2010/")
        print("   2. Search the act name")
        print("   3. Save PDF to data/raw/ folder")
    else:
        print("🎉 All laws downloaded successfully!")

    print(f"\n📂 Saved to: {Path(save_dir).absolute()}")


if __name__ == "__main__":
    download_laws()