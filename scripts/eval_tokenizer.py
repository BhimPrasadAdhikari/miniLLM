"""
Evaluate our BPE tokenizer's compression ratio against GPT-2 and GPT-4.

Metrics:
  - tokens produced for each text sample
  - bytes / tokens  →  compression ratio  (higher = better)
  - relative improvement over GPT-2 / GPT-4 baselines

Usage (from project root):
    python -m scripts.eval_tokenizer
"""

import sys
import os
from pathlib import Path

# Allow running as a plain script as well as a module
if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tiktoken
from tokenizer.tokenizer import Tokenizer


class TiktokenAdapter:
    """
    Thin wrapper around a tiktoken Encoding so its .encode() / .decode()
    signatures match our Tokenizer: no keyword-argument surprises.
    We allow all special tokens so the train-snippet (which contains
    '<|endoftext|>' separators) is encoded without errors.
    """

    def __init__(self, enc: tiktoken.Encoding):
        self._enc = enc
        self.n_vocab = enc.n_vocab

    def encode(self, text: str):
        return self._enc.encode(text, allowed_special="all")

    def decode(self, ids):
        return self._enc.decode(ids)

# ── ANSI colours for terminal output ──────────────────────────────────────
GREEN = "\033[92m"
RED   = "\033[91m"
RESET = "\033[0m"

# ── Diverse text samples that stress-test different token distributions ────

# English news excerpt (real-world proper nouns, numbers, punctuation)
news_text = r"""
(Washington, D.C., July 9, 2025)- Yesterday, Mexico's National Service of Agro-Alimentary Health, Safety, and Quality (SENASICA) reported a new case of New World Screwworm (NWS) in Ixhuatlan de Madero, Veracruz in Mexico, which is approximately 160 miles northward of the current sterile fly dispersal grid, on the eastern side of the country and 370 miles south of the U.S./Mexico border. This new northward detection comes approximately two months after northern detections were reported in Oaxaca and Veracruz, less than 700 miles away from the U.S. border, which triggered the closure of our ports to Mexican cattle, bison, and horses on May 11, 2025.

While USDA announced a risk-based phased port re-opening strategy for cattle, bison, and equine from Mexico beginning as early as July 7, 2025, this newly reported NWS case raises significant concern about the previously reported information shared by Mexican officials and severely compromises the outlined port reopening schedule of five ports from July 7-September 15. Therefore, in order to protect American livestock and our nation's food supply, Secretary Rollins has ordered the closure of livestock trade through southern ports of entry effective immediately.

"The United States has promised to be vigilant — and after detecting this new NWS case, we are pausing the planned port reopening's to further quarantine and target this deadly pest in Mexico. We must see additional progress combatting NWS in Veracruz and other nearby Mexican states in order to reopen livestock ports along the Southern border," said U.S. Secretary of Agriculture Brooke L. Rollins. "Thanks to the aggressive monitoring by USDA staff in the U.S. and in Mexico, we have been able to take quick and decisive action to respond to the spread of this deadly pest."
""".strip()

# Non-English text — tests whether our tokenizer handles CJK efficiently
korean_text = r"""
정직한 사실 위에, 공정한 시선을 더하다
Herald Korea Times

헤럴드코리아타임즈는 정치, 경제, 사회, 문화 등 한국 사회 전반의 주요 이슈를 심도 있게 다루는 종합 온라인 신문사입니다.

우리는 단순히 뉴스를 전달하는 것이 아니라, 사실(Fact)에 기반한 양측의 시각을 균형 있게 조명하며, 독자 여러분이 스스로 판단할 수 있는 '정보의 균형'을 제공합니다.

한국 언론의 오랜 문제로 지적되어 온 정치적 편향, 이념적 왜곡에서 벗어나
오직 정직함과 공정함을 원칙으로 삼는 언론을 지향합니다.
어느 한쪽의 주장만을 확대하거나 감추지 않고,
**모든 쟁점에 대해 '무엇이 쟁점인지', '누가 무엇을 주장하는지', '사실은 무엇인지'**를 명확히 전달하는 데 집중합니다.
""".strip()

# Source code — identifier-heavy, lots of indentation and symbols
code_text = r"""
class BasicTokenizer(Tokenizer):

    def __init__(self):
        super().__init__()

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # input text preprocessing
        text_bytes = text.encode("utf-8") # raw bytes
        ids = list(text_bytes) # list of integers in range 0..255

        # iteratively merge the most common pairs to create new tokens
        merges = {} # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)} # int -> bytes
        for i in range(num_merges):
            # count up the number of times every consecutive pair appears
            stats = get_stats(ids)
            # find the pair with the highest count
            pair = max(stats, key=stats.get)
            # mint a new token: assign it the next available id
            idx = 256 + i
            # replace all occurrences of pair in ids with idx
            ids = merge(ids, pair, idx)
            # save the merge
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            # prints
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")
""".strip()

# LaTeX mathematics — dense with symbols and backslash commands
math_text = r"""
\documentclass[12pt]{article}
\usepackage{amsmath,amsthm,amssymb}
\usepackage[margin=1in]{geometry}

\newtheorem{theorem}{Theorem}
\newtheorem*{remark}{Remark}

\begin{document}

\begin{center}
{\Large A Cute Identity: The Sum of Cubes is a Square}
\end{center}

\begin{theorem}
For every integer $n \ge 1$,
\[
\sum_{k=1}^{n} k^{3} \;=\; \left(\frac{n(n+1)}{2}\right)^{2}.
\]
\end{theorem}

\begin{proof}[Proof 1 (Induction)]
Let $S(n) = \sum_{k=1}^{n} k^3$. For $n=1$, $S(1)=1=(1\cdot 2/2)^2$, so the base case holds.

Assume $S(n)=\big(\tfrac{n(n+1)}{2}\big)^2$ for some $n\ge 1$.
Then
\[
S(n+1)
= S(n) + (n+1)^3
= \left(\frac{n(n+1)}{2}\right)^2 + (n+1)^3.
\]
Factor out $(n+1)^2$:
\[
S(n+1)
= (n+1)^2\left( \frac{n^2}{4} + (n+1) \right)
= (n+1)^2\left( \frac{n^2 + 4n + 4}{4} \right)
= (n+1)^2\left( \frac{(n+2)^2}{4} \right).
\]
Thus
\[
S(n+1)=\left(\frac{(n+1)(n+2)}{2}\right)^{2},
\]
which matches the claimed formula with $n$ replaced by $n+1$. By induction, the identity holds for all $n\ge 1$.
\end{proof}

\begin{proof}[Proof 2 (Algebraic telescoping)]
Recall the binomial identity
\[
(k+1)^4 - k^4 = 4k^3 + 6k^2 + 4k + 1.
\]
Summing both sides from $k=0$ to $n$ telescopes:
\[
(n+1)^4 - 0^4
= \sum_{k=0}^{n}\big(4k^3 + 6k^2 + 4k + 1\big)
= 4\sum_{k=1}^{n}k^3 + 6\sum_{k=1}^{n}k^2 + 4\sum_{k=1}^{n}k + (n+1).
\]
Using the standard sums
\[
\sum_{k=1}^{n}k = \frac{n(n+1)}{2}
\quad\text{and}\quad
\sum_{k=1}^{n}k^2 = \frac{n(n+1)(2n+1)}{6},
\]
solve for $\sum_{k=1}^{n}k^3$ to get
\[
\sum_{k=1}^{n}k^3 = \left(\frac{n(n+1)}{2}\right)^2.
\]
\end{proof}

\begin{remark}
Geometrically, the identity says: ``adding up $1^3,2^3,\dots,n^3$ builds a perfect square''—namely the square of the $n$th triangular number. This is why one sometimes calls it the \emph{sum-of-cubes is a square} phenomenon.
\end{remark}

\end{document}
""".strip()

# Technical scientific prose — long compound words, subscripts represented as text
science_text = r"""
Photosynthesis is a photochemical energy transduction process in which light-harvesting pigment–protein complexes within the thylakoid membranes of oxygenic phototrophs absorb photons and initiate charge separation at the reaction center, driving the linear electron transport chain from water to NADP⁺ via photosystem II, the cytochrome b₆f complex, and photosystem I, concomitantly generating a trans-thylakoid proton motive force utilized by chloroplastic ATP synthase. The light-dependent reactions produce ATP and NADPH, which fuel the Calvin–Benson–Bassham cycle in the stroma, wherein ribulose-1,5-bisphosphate is carboxylated by ribulose-1,5-bisphosphate carboxylase/oxygenase (RuBisCO) to form 3-phosphoglycerate, subsequently reduced and regenerated through a series of enzymatic steps, enabling net assimilation of CO₂ into triose phosphates and ultimately carbohydrates. This process is tightly regulated by photoprotective mechanisms, redox feedback, and metabolite flux, representing a central biochemical pathway coupling solar energy capture to the biosphere's primary productivity.
""".strip()

# All samples collected in (name, text) order for consistent table rows
all_samples = [
    ("news",    news_text),
    ("korean",  korean_text),
    ("code",    code_text),
    ("math",    math_text),
    ("science", science_text),
]

# Optionally include a slice of the training corpus to see in-domain performance
TRAIN_FILE = Path("tokenizer_train.txt")
if TRAIN_FILE.exists():
    with open(TRAIN_FILE, encoding="utf-8") as fh:
        # Use the first ~50 KB so it runs quickly
        train_snippet = fh.read(50_000).rsplit("\n", 1)[0]
    all_samples.append(("train-snippet", train_snippet))


# ── Load tokenizers ────────────────────────────────────────────────────────

def load_tokenizers():
    """Return a dict {name: tokenizer_object} with a unified .encode(text) API."""

    tokenizers = {}

    # GPT-2 tokenizer via tiktoken (vocab ≈ 50 257) – wrapped so .encode() needs no kwargs
    tokenizers["gpt2"]  = TiktokenAdapter(tiktoken.get_encoding("gpt2"))

    # GPT-4 tokenizer via tiktoken (cl100k_base, vocab ≈ 100 277)
    tokenizers["gpt4"]  = TiktokenAdapter(tiktoken.get_encoding("cl100k_base"))

    # Our trained BPE tokenizer
    tokenizers["ours"]  = Tokenizer.load("tokenizer.json")

    return tokenizers


# ── Core metric computation ────────────────────────────────────────────────

def measure(tokenizer, text: str) -> dict:
    """
    Encode *text* and return a stats dict:
      bytes   – raw UTF-8 byte count
      tokens  – number of token ids produced
      ratio   – bytes / tokens  (compression ratio, higher = more efficient)
    """
    raw_bytes = text.encode("utf-8")

    # tiktoken and our Tokenizer both expose .encode(text) → list[int]
    token_ids = tokenizer.encode(text)

    return {
        "bytes":  len(raw_bytes),
        "tokens": len(token_ids),
        "ratio":  len(raw_bytes) / len(token_ids),
    }


def run_all(tokenizers: dict, samples: list) -> dict:
    """Return nested dict: results[tokenizer_name][sample_name] = stats."""
    results = {}
    for tok_name, tok in tokenizers.items():
        results[tok_name] = {}
        for sample_name, text in samples:
            results[tok_name][sample_name] = measure(tok, text)
    return results


# ── Pretty-print helpers ───────────────────────────────────────────────────

def color_better(val_a, val_b, fmt):
    """
    Return ANSI-coloured formatted strings for two values.
    The HIGHER value (better compression) gets GREEN, the lower gets RED.
    """
    s_a = format(val_a, fmt)
    s_b = format(val_b, fmt)
    if val_a > val_b:
        return f"{GREEN}{s_a}{RESET}", f"{RED}{s_b}{RESET}"
    elif val_b > val_a:
        return f"{RED}{s_a}{RESET}", f"{GREEN}{s_b}{RESET}"
    return s_a, s_b   # tie – no colour


def print_comparison(baseline_name: str, baseline_res: dict, ours_res: dict, samples: list):
    """
    Print a side-by-side table comparing baseline vs ours for every sample.
    Positive relative diff means our tokenizer uses fewer tokens (better).
    """
    COL = 95
    print(f"\nComparison with {baseline_name}:")
    print("=" * COL)
    print(f"{'Text':<16} {'Bytes':<8} "
          f"{baseline_name+' Tok':<8} {baseline_name+' Ratio':<12} "
          f"{'Ours Tok':<10} {'Ours Ratio':<12} "
          f"{'Rel Diff':<10} {'Better'}")
    print("-" * COL)

    for name, _ in samples:
        b = baseline_res[name]   # baseline stats
        o = ours_res[name]       # our stats

        # Relative token reduction: positive → we use fewer tokens
        rel = (b["tokens"] - o["tokens"]) / b["tokens"] * 100

        # Colour ratios: higher ratio = better compression
        c_b_ratio, c_o_ratio = color_better(b["ratio"], o["ratio"], ".2f")
        # Colour token counts: fewer tokens = better (invert the comparison)
        c_b_tok,   c_o_tok   = color_better(o["tokens"], b["tokens"], "d")  # note: swapped so lower gets green
        c_b_tok,   c_o_tok   = c_o_tok, c_b_tok  # swap back to correct columns

        # Colour the relative diff sign
        diff_str = f"{rel:+.1f}%"
        if rel > 0:
            diff_str = f"{GREEN}{diff_str}{RESET}"
        elif rel < 0:
            diff_str = f"{RED}{diff_str}{RESET}"

        better = "Ours" if o["ratio"] > b["ratio"] else (baseline_name if b["ratio"] > o["ratio"] else "Tie")

        print(f"{name:<16} {b['bytes']:<8} "
              f"{c_b_tok:<8} {c_b_ratio:<12} "
              f"{c_o_tok:<10} {c_o_ratio:<12} "
              f"{diff_str:<10} {better}")

    print("=" * COL)


# ── Entry point ────────────────────────────────────────────────────────────

def main():
    print("Loading tokenizers...")
    tokenizers = load_tokenizers()

    # Show vocab sizes so the reader knows who they're comparing against
    print(f"\nVocab sizes:")
    print(f"  GPT-2 : {tokenizers['gpt2'].n_vocab:,}")
    print(f"  GPT-4 : {tokenizers['gpt4'].n_vocab:,}")
    print(f"  Ours  : {len(tokenizers['ours'].vocab):,}  (32 000 merges + 256 bytes + 1 special)")

    print("\nRunning compression benchmarks…")
    results = run_all(tokenizers, all_samples)

    # Print one comparison table per baseline
    print_comparison("GPT-2", results["gpt2"], results["ours"], all_samples)
    print_comparison("GPT-4", results["gpt4"], results["ours"], all_samples)

    # Quick round-trip sanity check – encode then decode must be lossless
    print("\nRound-trip sanity check (our tokenizer):")
    for name, text in all_samples:
        ids    = tokenizers["ours"].encode(text)
        back   = tokenizers["ours"].decode(ids)
        status = "PASS" if back == text else "FAIL"
        print(f"  {name:<16} {status}  ({len(ids)} tokens, {results['ours'][name]['ratio']:.2f}x compression)")


if __name__ == "__main__":
    main()
