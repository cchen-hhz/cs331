import sys
import os
import numpy as np
from typing import List, Generator

from cs336_basics.boundary_chunk import find_chunk_boundaries
from cs336_basics.tokenizer import tokenizer

if len(sys.argv) <= 1:
    print(f"Missing file name!")
    sys.exit(-1)

file_dir = "modeler/dist/data"
file_path = f"{file_dir}/{sys.argv[1]}.txt"
desire_size = 1024 * 1024 * 5

# Rename instance to avoid shadowing class
enc = tokenizer.fromfiles("modeler/dist/bpe/vocab.pkl",
                          "modeler/dist/bpe/merges.pkl",
                          special_tokens=["<|endoftext|>"])

def gen_intervals(path: str, intervals: List[int]) -> Generator[str, None, None]:
    with open(path, 'rb') as f:
        for i in range(len(intervals) - 1):
            f.seek(intervals[i])
            length = intervals[i + 1] - intervals[i]
            if length > 0:
                chunk = f.read(length)
                yield chunk.decode('utf-8', errors='ignore')

try:
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        sys.exit(-1)

    with open(file_path, 'rb') as f:
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        num_chunks = max(1, file_size // desire_size)
        intervals = find_chunk_boundaries(f, num_chunks, b"<|endoftext|>")

    out_file = f"{file_dir}/{sys.argv[1]}.bin"
    print(f"Writing to {out_file}...")
    
    # Determine dtype based on vocabulary size
    dtype = np.uint16 if len(enc.vocab) < 65536 else np.uint32
    
    buffer = []
    BUFFER_FLUSH_SIZE = 1024 * 1024 * 10

    with open(out_file, "wb") as w:
        # Use enc.encode_iterable with the generator directly
        for token in enc.encode_iterable(gen_intervals(file_path, intervals)):
            buffer.append(token)
            if len(buffer) >= BUFFER_FLUSH_SIZE:
                np.array(buffer, dtype=dtype).tofile(w)
                buffer = []
        
        # Flush remaining tokens
        if buffer:
            np.array(buffer, dtype=dtype).tofile(w)
            
    print(f"Done! Saved with dtype {dtype}.")

except Exception as e:
    print(f"Error when handling!!! {e}")
    import traceback
    traceback.print_exc()