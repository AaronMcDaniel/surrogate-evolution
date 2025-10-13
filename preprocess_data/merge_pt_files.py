import torch

merged = {}
offset = 0

for i in range(1, 51):
    path = f"/home/hice1/wlu314/scratch/surrogate-evolution/preprocess_data/data_with_encoding_{i}.pt"
    chunk = torch.load(path, map_location="cpu")  # each is a dict: {idx: (info, datum, label), …}

    # copy into merged, shifting keys by the current offset
    for local_idx, sample in chunk.items():
        merged[offset + local_idx] = sample

    offset += len(chunk)  # bump offset for the next file
    print(f"Merged {path}  →  current total samples: {offset}")

# finally, write out the combined dict
torch.save(merged, "data_with_encoding_all.pt")
print(f"All done! Saved {len(merged)} samples to data_with_encoding_all.pt")
