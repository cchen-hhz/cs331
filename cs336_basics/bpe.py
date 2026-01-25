from collections import Counter, defaultdict
from typing import BinaryIO, List, Tuple, Dict, Set, Optional
import multiprocessing
from cs336_basics.boundary_chunk import find_chunk_boundaries
import regex as re
import heapq
import os

class preTokenizer:
    def __init__(self, special_tokens: List[str]):
        self.special_tokens = {token: token.encode('utf-8') for token in special_tokens}
        self.word_pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        if special_tokens:
            special_token_pat = "|".join(token for token in self.special_tokens.keys())
            self.special_pat = re.compile(special_token_pat)
        else:
            self.special_pat = None

    def tokenizer(self, text: str):
        if not self.special_pat:
            for match in self.word_pat.finditer(text):
                yield match.group().encode("utf-8")
            return
        
        chunks = self.special_pat.split(text)

        for chunk in chunks:
            if not chunk:
                continue

            if chunk in self.special_tokens:
                yield self.special_tokens[chunk]
            else:
                for match in self.word_pat.finditer(chunk):
                    yield match.group().encode("utf-8")


class BPETrainer:
    def __init__(self, vocab_size: int, special_tokens: List[str]):
        self.require_vocab_size = vocab_size
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.merge: List[Tuple[bytes, bytes]] = []
        self.preTokenizer = preTokenizer(special_tokens)
        
        for special in special_tokens:
            encoded = special.encode("utf-8")
            self.vocab[len(self.vocab)] = encoded

    def bpe_train(self, text: str) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
        tokens = self.preTokenizer.tokenizer(text)
        special_tokens = set(self.preTokenizer.special_tokens.values())
        word_freq = Counter(token for token in tokens if token not in special_tokens)
        return self.train_from_counts(word_freq)

    def train_from_counts(self, word_freq: Counter) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
        splits: Dict[bytes, List[bytes]] = {word: [bytes([b]) for b in word] for word in word_freq}
        pair_positions: Dict[Tuple[bytes, bytes], Set[bytes]] = defaultdict(set)
        pair_freqs: Dict[Tuple[bytes, bytes], int] = defaultdict(int)
        heap: List[Tuple[int, Tuple[bytes, bytes]]]= []
        merges: List[Tuple[bytes, bytes]] = []

        for word, freq in word_freq.items():
            split = splits[word]
            if len(split) > 1:
                for i, j in zip(split[:-1], split[1:]):
                    pair = (i, j)
                    pair_freqs[pair] += freq
                    pair_positions[pair].add(word)
        
        for pair, freq in pair_freqs.items():
            heapq.heappush(heap, (-freq, pair))

        print(f"init done, start training")
        while len(self.vocab) < self.require_vocab_size:
            best_pair = self._find_best_pair(heap, pair_freqs)
             
            if best_pair == None:
                break

            merges.append(best_pair)
            new_token = best_pair[0] + best_pair[1]
            self.vocab[len(self.vocab)] = new_token

            # update
            for token in pair_positions.get(best_pair, []):
                freq = word_freq[token]
                piece = splits[token]

                i = 0
                while i < len(piece) - 1:
                    if piece[i] == best_pair[0] and piece[i + 1] == best_pair[1]:
                        piece[i] = new_token
                        piece.pop(i+1)

                        if i > 0:
                            old_pair = (piece[i-1], best_pair[0])
                            cur_pair = (piece[i-1], new_token)
                            self._updator(token, old_pair, -freq, pair_freqs, pair_positions, heap)
                            self._updator(token, cur_pair, freq, pair_freqs, pair_positions, heap)

                        if i < len(piece) - 1:
                            old_pair = (best_pair[1], piece[i+1])
                            cur_pair = (new_token, piece[i+1])
                            self._updator(token, old_pair, -freq, pair_freqs, pair_positions, heap)
                            self._updator(token, cur_pair, freq, pair_freqs, pair_positions, heap)
                            
                    else:
                        i += 1
            if best_pair in pair_freqs:
                del pair_freqs[best_pair]
            if best_pair in pair_positions:
                del pair_positions[best_pair]
        
        return self.vocab, merges
        
    def _updator(self, token, pair, delta, pair_freq, pair_position, heap):
        pair_freq[pair] += delta
        if pair_freq[pair] > 0:
            pair_position[pair].add(token)
            heapq.heappush(heap, (-pair_freq[pair], pair))
        else:
            del pair_freq[pair]
            if pair in pair_position:
                pair_position[pair].discard(token)
                if not pair_position[pair]:
                    del pair_position[pair]

    def _find_best_pair(self, heap: List, pair_freq: Dict) -> Optional[Tuple[bytes, bytes]]:
        while pair_freq:
            freq, pair = heapq.heappop(heap)
            freq = -freq

            if pair not in pair_freq or freq != pair_freq[pair]:
                continue
        
            best_pair = pair
            res = []
            while heap and heap[0][0] == -freq:
                _, new_pair = heapq.heappop(heap)

                if new_pair not in pair_freq or freq != pair_freq[new_pair]:
                    continue

                if new_pair > best_pair:
                    res.append(best_pair)
                    best_pair = new_pair
                else:
                    res.append(new_pair)
            
            for pair in res:
                heapq.heappush(heap, (-freq, pair))
        
            return best_pair
        return None




def _process_chunk(args):
    input_path, start, end, special_tokens = args
    pt = preTokenizer(special_tokens)
    with open(input_path, 'rb') as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")
    
    tokens = pt.tokenizer(text)
    special_tokens_set = set(pt.special_tokens.values())
    return Counter(token for token in tokens if token not in special_tokens_set)

def bpe_train(
        input_path: str | os.PathLike,
        vocab_size: int,
        speial_tokens: List[str]
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    num_process = 10

    split_token = b'<|endoftext|>'
    if speial_tokens:
        split_token = speial_tokens[0].encode("utf-8")
    
    with open(input_path, 'rb') as f:
        boundaries = find_chunk_boundaries(f, num_process, split_token)

    chunk_args = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        chunk_args.append((input_path, start, end, speial_tokens))

    print(f"Starting {num_process} processes...")
    word_freq = Counter()
    with multiprocessing.Pool(processes=num_process) as pool:
        for c in pool.imap_unordered(_process_chunk, chunk_args):
            word_freq.update(c)
    
    print(f"freq done")

    trainer = BPETrainer(vocab_size, speial_tokens)
    vocab, merges = trainer.train_from_counts(word_freq)
    return vocab, merges

    
    