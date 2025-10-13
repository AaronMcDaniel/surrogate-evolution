import pandas as pd

def split_csv(input_file: str, chunk_size: int = 1, output_prefix: str = "data_"):
    """
    Reads `input_file` in chunks of `chunk_size` rows and writes each chunk
    (including header) to a separate CSV file:
      data_1.csv, data_2.csv, ..., data_n.csv
    """
    for i, chunk in enumerate(pd.read_csv(input_file, chunksize=chunk_size), start=1):
        out_name = f"{output_prefix}{i}.csv"
        chunk.to_csv(out_name, index=False)
        print(f"Wrote {out_name} ({len(chunk)} rows)")

if __name__ == "__main__":
    split_csv("compiled_data_valid_only.csv", chunk_size=1, output_prefix="data_")
