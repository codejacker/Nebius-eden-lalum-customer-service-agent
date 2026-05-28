"""Downloads the Bitext Customer Service dataset from HuggingFace to data/bitext.csv."""
from pathlib import Path


def download():
    output = Path("data/bitext.csv")

    if output.exists():
        size_kb = output.stat().st_size // 1024
        print(f"Dataset already exists at {output} ({size_kb} KB). Nothing to do.")
        return

    print("Downloading Bitext dataset from HuggingFace (one-time, ~20MB)...")
    from datasets import load_dataset

    ds = load_dataset(
        "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
        split="train",
    )
    df = ds.to_pandas()
    df.to_csv(output, index=False)
    print(f"Saved {len(df):,} rows → {output}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Categories: {sorted(df['category'].unique().tolist())}")


if __name__ == "__main__":
    download()
