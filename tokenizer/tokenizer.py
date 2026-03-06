import re
import json
import regex
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


class Tokenizer:
    """
    Byte-Pair Encoding (BPE) tokenizer – trained from scratch.

    Workflow:
        1. Call .train(text, num_merges) to build the vocabulary.
        2. Call .add_special_tokens([...]) to register special tokens.
        3. Use .encode(text) / .decode(ids) for inference.
        4. Persist with .save(path) / Tokenizer.load(path).
    """

    def __init__(self):
        # Base vocab: every possible byte  (256 entries)
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.merges: Dict[Tuple[int, int], int] = {}

        # Special tokens:  string → id   and   id → string
        self.special_tokens: Dict[str, int] = {}
        self._special_id_to_str: Dict[int, str] = {}

        # GPT-4 / tiktoken-compatible pre-tokenisation pattern
        self._pre_token_pattern = regex.compile(
            r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
        )

    # ── Special tokens ─────────────────────────────────────────────────────

    def add_special_tokens(self, tokens: List[str]) -> None:
        """Register special tokens. IDs are assigned after all merge IDs."""
        for token in tokens:
            if token not in self.special_tokens:
                new_id = 256 + len(self.merges) + len(self.special_tokens)
                self.special_tokens[token] = new_id
                self._special_id_to_str[new_id] = token
                self.vocab[new_id] = token.encode("utf-8")

    # ── Pre-tokenisation ───────────────────────────────────────────────────

    def _pre_tokenize(self, text: str) -> List[str]:
        return self._pre_token_pattern.findall(text)

    # ── Training ───────────────────────────────────────────────────────────

    def train(self, text: str, num_merges: int = 1000, verbose: bool = True) -> None:
        """
        Run BPE on *text* for *num_merges* merge steps.

        Args:
            text:       Raw training corpus.
            num_merges: How many merge rules to learn (= extra vocab tokens).
            verbose:    Print progress every 100 merges.
        """
        if not text.strip():
            raise ValueError("Training text cannot be empty")

        chunks = self._pre_tokenize(text)
        if verbose:
            print(f"Pre-tokenized into {len(chunks):,} chunks")

        # Build frequency table over pre-token byte sequences
        word_freq: Dict[Tuple[int, ...], int] = defaultdict(int)
        for chunk in chunks:
            if chunk:
                word_freq[tuple(chunk.encode("utf-8"))] += 1

        if verbose:
            print(f"Unique pre-token types: {len(word_freq):,}")

        for i in range(num_merges):
            # Count adjacent-pair frequencies
            pair_freq: Dict[Tuple[int, int], int] = defaultdict(int)
            for word, freq in word_freq.items():
                for j in range(len(word) - 1):
                    pair_freq[(word[j], word[j + 1])] += freq

            if not pair_freq:
                if verbose:
                    print(f"No more pairs. Stopped after {i} merges.")
                break

            best_pair = max(pair_freq, key=pair_freq.get)
            new_id    = 256 + len(self.merges)

            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            # Apply the merge to all words
            new_word_freq: Dict[Tuple[int, ...], int] = defaultdict(int)
            for word, freq in word_freq.items():
                new_word = []
                j = 0
                while j < len(word):
                    if j < len(word) - 1 and (word[j], word[j + 1]) == best_pair:
                        new_word.append(new_id)
                        j += 2
                    else:
                        new_word.append(word[j])
                        j += 1
                new_word_freq[tuple(new_word)] += freq
            word_freq = new_word_freq

            if verbose and (i + 1) % 100 == 0:
                print(
                    f"  Merge {i+1:4d}/{num_merges}: {best_pair} → {new_id} "
                    f"(freq={pair_freq[best_pair]})"
                )

        if verbose:
            print(f"Training complete. Vocab size: {len(self.vocab)}")

    # ── Encode ─────────────────────────────────────────────────────────────

    def encode(self, text: str) -> List[int]:
        """Encode *text* to a list of token ids."""
        if self.special_tokens:
            return self._encode_with_special(text)
        return self._encode_chunk(text)

    def _encode_chunk(self, text: str) -> List[int]:
        """Encode a single string that contains no special tokens."""
        chunks = self._pre_tokenize(text)
        result: List[int] = []

        # Lookup table: (a, b) → merge rank (lower = higher priority)
        merge_rank: Dict[Tuple[int, int], int] = {
            pair: rank for rank, pair in enumerate(self.merges.keys())
        }

        for chunk in chunks:
            if not chunk:
                continue

            ids: List[int] = list(chunk.encode("utf-8"))

            # Greedily apply the highest-priority available merge
            while len(ids) >= 2:
                best_rank = float("inf")
                best_idx  = -1

                for i in range(len(ids) - 1):
                    pair = (ids[i], ids[i + 1])
                    rank = merge_rank.get(pair, float("inf"))
                    if rank < best_rank:
                        best_rank = rank
                        best_idx  = i

                if best_idx == -1 or best_rank == float("inf"):
                    break  # fully merged

                pair   = (ids[best_idx], ids[best_idx + 1])
                new_id = self.merges[pair]
                ids    = ids[:best_idx] + [new_id] + ids[best_idx + 2:]

            result.extend(ids)

        return result

    def _encode_with_special(self, text: str) -> List[int]:
        """Split on special tokens first, then encode normal parts."""
        pattern = "(" + "|".join(
            re.escape(tok)
            for tok in sorted(self.special_tokens.keys(), key=len, reverse=True)
        ) + ")"
        result = []
        for part in re.split(pattern, text):
            if part in self.special_tokens:
                result.append(self.special_tokens[part])
            elif part:
                result.extend(self._encode_chunk(part))
        return result

    # ── Decode ─────────────────────────────────────────────────────────────

    def decode(self, ids: List[int]) -> str:
        """Decode a list of token ids back to a string."""
        byte_pieces = []
        for tid in ids:
            if tid in self._special_id_to_str:
                byte_pieces.append(self._special_id_to_str[tid].encode("utf-8"))
            else:
                byte_pieces.append(self.vocab.get(tid, bytes([tid & 0xFF])))
        return b"".join(byte_pieces).decode("utf-8", errors="replace")

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Serialize the tokenizer to a JSON file."""
        data = {
            "merges": [[list(pair), new_id] for pair, new_id in self.merges.items()],
            "special_tokens": self.special_tokens,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Tokenizer saved → {path}")

    @classmethod
    def load(cls, path: str) -> "Tokenizer":
        """Deserialize a tokenizer from a JSON file."""
        tok = cls()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for pair_list, new_id in data["merges"]:
            pair = (pair_list[0], pair_list[1])
            tok.merges[pair] = new_id
            tok.vocab[new_id] = tok.vocab[pair[0]] + tok.vocab[pair[1]]

        for token, tid in data["special_tokens"].items():
            tok.special_tokens[token]   = tid
            tok._special_id_to_str[tid] = token
            tok.vocab[tid]              = token.encode("utf-8")

        print(
            f"Tokenizer loaded ← {path}  "
            f"({len(tok.merges)} merges, vocab={len(tok.vocab)})"
        )
        return tok
