import pandas as pd
from pathlib import Path

def run_ingest():
    # It looks in the current folder (.)
    current_dir = Path(".") 
    output_file = Path("ncua_data.parquet")
    
    # Let's start with the two most important files
    files = {"FS220.txt": "financials", "FOICU.txt": "profiles"}
    dfs = []

    for filename in files.keys():
        p = current_dir / filename
        if p.exists():
            df = pd.read_csv(p, sep="|", encoding="latin-1", low_memory=False)
            df.columns = [c.lower().strip() for c in df.columns]
            dfs.append(df)
            print(f"✅ Loaded {filename}")
        else:
            print(f"❌ Could not find {filename} in this folder.")

    if len(dfs) == 2:
        # Merge them on the Charter Number
        final_df = pd.merge(dfs[0], dfs[1], on="cu_number", how="inner")
        final_df.to_parquet(output_file)
        print(f"✨ SUCCESS! Created {output_file}")
    else:
        print("Ensure FS220.txt and FOICU.txt are in this folder.")

if __name__ == "__main__":
    run_ingest()
