from cs336_basics.tokenizer import tokenizer

with open("cs336_basics/data/owt-cut.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokenizer = tokenizer.fromfiles(
    "cs336_basics/data/vocab.pkl",
    "cs336_basics/data/merges.pkl",
    ["<|endoftext|>"])

name = "owt-cut"
last_len = len(text.encode("utf-8"))
cur_len = len(tokenizer.encode(text))
print(f"compare for {name}: {last_len} and {cur_len}, rate {cur_len / last_len}")