from cs336_basics.bpe import bpe_train
import pickle

vocab, merges = bpe_train(
    "./data/TinyStoriesV2-GPT4-train.txt",
    10000,
    ["<|endoftext|>"])

with open("vocab.json", 'wb') as f:
    pickle.dump(vocab, f)

with open("merges.json", 'wb') as f:
    pickle.dump(merges, f)