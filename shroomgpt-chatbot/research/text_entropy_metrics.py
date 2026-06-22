"""
text_entropy_metrics.py
=======================

Entropy/complexity signatures computed on the OUTPUT TEXT AS A SIGNAL -- the
analogue of the 2021 Entropy study computing complexity on the EEG/DeepDream
signal, NOT on the model's internal processing.

Four complexity signatures + one coherence guardrail:

  1. token_lzc       -- Lempel-Ziv (LZ76) complexity of the token stream.
                        Direct port of the Schartner et al. (2017) EEG measure.
  2. token_entropy   -- Shannon entropy of the empirical unigram + bigram
                        distribution of the output. The "signal entropy" analogue.
  3. distinct2/3     -- fraction of unique bi/tri-grams (lexical diversity).
  4. semantic_*      -- the CONNECTIVITY analogue (Viol et al. 2017): embed each
                        sentence, measure (a) mean cosine DISTANCE between
                        consecutive sentences = associative looseness, and
                        (b) spread of the sentence-embedding cloud = semantic
                        volume explored. Higher under psychedelics.

  coherence (guardrail) -- the model's own perplexity on its output. EVERY
                        complexity metric above is maximized by word salad, so
                        results are only meaningful read AGAINST coherence:
                        the target is "high complexity at acceptable coherence",
                        not "high complexity period".

The semantic metrics need sentence-transformers (small, all-MiniLM-L6-v2 ~80MB).
If it's unavailable, those fields come back None and the rest still work.
"""

import math
import re
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import torch


# ----------------------------- LZ76 -----------------------------

def _lz76(arr):
    if isinstance(arr, np.ndarray):
        buf = arr.astype(np.int64).tolist()
    else:
        buf = list(arr)
    n = len(buf)
    if n <= 1:
        return n
    i, l, c, k, k_max = 0, 1, 1, 1, 1
    while True:
        if l + k > n:
            c += 1
            break
        if buf[i + k - 1] != buf[l + k - 1]:
            if k > k_max:
                k_max = k
            i += 1
            if i == l:
                c += 1
                l += k_max
                if l >= n:
                    break
                i = 0; k = 1; k_max = 1
            else:
                k = 1
        else:
            k += 1
    return c


def token_lzc(token_ids):
    ids = list(token_ids)
    n = len(ids)
    if n < 2:
        return 0.0
    uniq = {t: i for i, t in enumerate(dict.fromkeys(ids))}
    compact = [uniq[t] for t in ids]
    c = _lz76(compact)
    K = max(2, len(uniq))
    return c / (n / math.log(n, K)) if n > 1 else 0.0


# ----------------------- token entropy --------------------------

def _shannon(counts):
    total = sum(counts.values())
    if total == 0:
        return 0.0
    h = 0.0
    for v in counts.values():
        p = v / total
        h -= p * math.log(p, 2)
    return h


def token_entropy(token_ids):
    """Mean of normalized unigram and bigram Shannon entropy (bits, 0..1)."""
    ids = list(token_ids)
    n = len(ids)
    if n < 2:
        return 0.0
    from collections import Counter
    uni = Counter(ids)
    big = Counter(zip(ids[:-1], ids[1:]))
    h_uni = _shannon(uni)
    h_big = _shannon(big)
    # normalize by max possible entropy for the realized support
    h_uni_n = h_uni / math.log(len(uni), 2) if len(uni) > 1 else 0.0
    h_big_n = h_big / math.log(len(big), 2) if len(big) > 1 else 0.0
    return 0.5 * (h_uni_n + h_big_n)


# --------------------- lexical diversity ------------------------

def distinct_n(token_ids, n):
    if len(token_ids) < n:
        return 0.0
    grams = [tuple(token_ids[i:i+n]) for i in range(len(token_ids) - n + 1)]
    return len(set(grams)) / len(grams)


# ------------------- semantic trajectory ------------------------

_SENT_MODEL = None

def _get_sentence_model():
    global _SENT_MODEL
    if _SENT_MODEL == "unavailable":
        return None
    if _SENT_MODEL is None:
        import os
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:
            print(f"[metrics] sentence-transformers not installed ({e}); "
                  f"semantic metrics disabled. Run: pip install sentence-transformers")
            _SENT_MODEL = "unavailable"
            return None
        # Try a sequence of load strategies so a flaky network doesn't disable
        # the metric when the model is already cached locally.
        last_err = None
        for attempt in ("online", "offline"):
            try:
                if attempt == "offline":
                    os.environ["HF_HUB_OFFLINE"] = "1"
                    os.environ["TRANSFORMERS_OFFLINE"] = "1"
                _SENT_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
                print(f"[metrics] sentence model loaded ({attempt}).")
                return _SENT_MODEL
            except Exception as e:
                last_err = e
        print(f"[metrics] could not load all-MiniLM-L6-v2 ({last_err}); "
              f"semantic metrics disabled. To cache it once while online:\n"
              f"    python -c \"from sentence_transformers import SentenceTransformer; "
              f"SentenceTransformer('all-MiniLM-L6-v2')\"")
        _SENT_MODEL = "unavailable"
        return None
    return _SENT_MODEL if _SENT_MODEL != "unavailable" else None


def _split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 0]


def semantic_trajectory(text):
    """
    Connectivity analogue. Returns (consecutive_jump, cloud_spread) or (None,None).
      consecutive_jump: mean cosine DISTANCE between consecutive sentence embeddings
                        (higher = looser association, bigger idea-to-idea leaps)
      cloud_spread:     mean pairwise cosine distance across all sentences
                        (higher = larger semantic volume explored)
    """
    model = _get_sentence_model()
    if model is None:
        return None, None
    sents = _split_sentences(text)
    if len(sents) < 2:
        return None, None
    emb = model.encode(sents, convert_to_numpy=True, normalize_embeddings=True)
    # consecutive jump
    cons = []
    for i in range(1, len(emb)):
        cons.append(1.0 - float(np.dot(emb[i], emb[i-1])))
    consecutive_jump = float(np.mean(cons))
    # cloud spread (mean pairwise distance)
    n = len(emb)
    sims = emb @ emb.T
    iu = np.triu_indices(n, k=1)
    cloud_spread = float(np.mean(1.0 - sims[iu]))
    return consecutive_jump, cloud_spread


# --------------------------- container --------------------------

@dataclass
class TextMetrics:
    token_lzc: float
    token_entropy: float
    distinct2: float
    distinct3: float
    semantic_jump: Optional[float]
    semantic_spread: Optional[float]
    perplexity: float          # coherence guardrail (lower = more coherent)
    n_tokens: int

    def as_dict(self):
        return asdict(self)


@torch.no_grad()
def coherence_perplexity(model, tok, full_input_ids, n_prompt_tokens, gen_token_ids):
    """
    Teacher-forced perplexity of the model on its OWN output -- the coherence
    guardrail. Measured with NO interventions active (clean read of how
    'surprising' the text is to the base model). Lower = more coherent.
    """
    out = model(full_input_ids, use_cache=False)
    logits = out.logits[0].float()
    seq_len = logits.shape[0]
    nll = []
    for t, tid in enumerate(gen_token_ids):
        pos = n_prompt_tokens + t - 1
        if 0 <= pos < seq_len and 0 <= tid < logits.shape[1]:
            logp = torch.log_softmax(logits[pos], dim=-1)
            nll.append(-logp[tid].item())
    return float(np.exp(np.mean(nll))) if nll else float("inf")


def compute_text_metrics(gen_token_ids, gen_text, *, perplexity=float("nan")):
    """All text-as-signal metrics for one generation."""
    jump, spread = semantic_trajectory(gen_text)
    return TextMetrics(
        token_lzc=token_lzc(gen_token_ids),
        token_entropy=token_entropy(gen_token_ids),
        distinct2=distinct_n(gen_token_ids, 2),
        distinct3=distinct_n(gen_token_ids, 3),
        semantic_jump=jump,
        semantic_spread=spread,
        perplexity=perplexity,
        n_tokens=len(gen_token_ids),
    )