from typing import Dict, List, Tuple, Iterable, Iterator
import regex as re
import pickle

class tokenizer:
    def __init__(self,
                 vocab: Dict[int, bytes],
                 merge: List[Tuple[bytes, bytes]],
                 special_tokens: List[str] | None = None):
        self.vocab = vocab
        self.from_vocab = {token: i for i, token in vocab.items()}
        self.merge = merge
        self.merge_dict = {merge: i for i, merge in enumerate(merge)}
        self.word_pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        
        if special_tokens is not None:
            special_tokens = sorted(special_tokens, key=len, reverse=True)
            self.special_token_bytes = {token: token.encode("utf-8") for token in special_tokens}
            special_pat = "("+"|".join(re.escape(token) for token in special_tokens)+")"
            self.special_pat = re.compile(special_pat)
            self.special_tokens= special_tokens
        else:
            self.special_pat = None
            self.special_token = {}
            self.special_tokens = []

    @classmethod
    def fromfiles(cls, 
                  vocab_filepath: str, 
                  merge_filepath: str, 
                  special_tokens: List[str] | None = None):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merge_filepath, "rb") as f:
            merge = pickle.load(f)
        return tokenizer(vocab, merge, special_tokens)
    
    def _pre_tokenizer(self, text: str):
        if not self.special_pat:
            for match in self.word_pat.finditer(text):
                yield match.group().encode("utf-8")
            return
        
        chunks = self.special_pat.split(text)
        for chunk in chunks:
            if not chunk:
                continue
            if chunk in self.special_tokens:
                yield self.special_token_bytes[chunk]
            else:
                for token in self.word_pat.finditer(chunk):
                    yield token.group().encode("utf-8")


    def _encode_per(self, token: bytes) -> List[int]:
        if token in self.from_vocab:
            return [self.from_vocab[token]]
        
        token_list = [bytes([b]) for b in token]
        result = []
        while True:
            best_merge = None
            merge_pri = -1
            i = 0
            while i < len(token_list) - 1:
                pair = (token_list[i], token_list[i+1])
                if pair in self.merge_dict:
                    if not best_merge or merge_pri > self.merge_dict[pair]:
                        best_merge = pair
                        merge_pri = self.merge_dict[pair]
                i += 1
            if not best_merge:
                break
        
            new_list = []
            new_token = best_merge[0] + best_merge[1]
            i = 0
            while i < len(token_list):
                if i < len(token_list) - 1 and (token_list[i], token_list[i+1]) == best_merge:
                    new_list.append(new_token)
                    i += 2
                else:
                    new_list.append(token_list[i])
                    i += 1
            token_list = new_list

        return [self.from_vocab[t] for t in token_list]

    def encode(self, text: str) -> List[int]:
        result = []
        for chunk in self._pre_tokenizer(text):
            result.extend(self._encode_per(chunk))

        return result


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            for token in self.encode(text):
                yield token
    
    def decode(self, ids: List[int]) -> str:
        return b''.join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")
