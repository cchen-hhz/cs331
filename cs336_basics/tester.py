from cs336_basics.bpe import bpe_train

vocab, merges = bpe_train(
    "./data/TinyStoriesV2-GPT4-train.txt",
    10000,
    ["<|endoftext|>"])

print(vocab)

print(merges)