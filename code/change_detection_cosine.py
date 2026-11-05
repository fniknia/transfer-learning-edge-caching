"""
Popularity change detection with cosine similarity, plus a request-rate check.

The request stream has two segments. The first uses the Zipf popularity ranking. The second
uses the same Zipf distribution over a permuted set of file IDs, which is a popularity shift.
Arrivals are Poisson with mean interarrival time 0.2 throughout.

Popularity detector (as in the notebook):
  - reference: normalized request counts over the first L1 = 1000 requests
  - current:   normalized request counts over the last L2 = 50 requests
  - similarity: cosine similarity between reference and current, averaged over the last 20 values
  - flag: mean similarity < 0.3 (after 50 requests)

Rate detector (as in the notebook):
  - moving average of the last 10 interarrival times
  - flag: |average - 0.2| > 0.05

Usage:
    python change_detection_cosine.py --seed 0 --out results
"""
import argparse
import os

import numpy as np

FILE_TYPES = 50
L1 = 1000        # reference window
L2 = 50          # current window
SIM_WINDOW = 20  # similarity averaging window
SIM_THRESHOLD = 0.3
RATE_WINDOW = 10
RATE_MEAN = 0.2
RATE_TOL = 0.05
SEG = 1000       # requests per segment


def zipf_probs(alpha=1.0):
    ranks = np.arange(1, FILE_TYPES + 1)
    p = 1.0 / ranks ** alpha
    return p / p.sum()


def make_stream(rng):
    """Segment 1: Zipf popularity. Segment 2: the same Zipf over a permuted file ID mapping."""
    p = zipf_probs()
    seg1 = rng.choice(np.arange(1, FILE_TYPES + 1), size=SEG, p=p)
    perm = rng.permutation(np.arange(1, FILE_TYPES + 1))
    seg2 = perm[rng.choice(np.arange(FILE_TYPES), size=SEG, p=p)]
    requests = np.concatenate([seg1, seg2])
    interarrivals = rng.exponential(RATE_MEAN, size=2 * SEG)
    return requests, interarrivals


def normalized_counts(requests, window):
    counts = np.bincount(requests - 1, minlength=FILE_TYPES).astype(float)
    return counts / window


def cosine_similarity(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def run(seed, out_dir):
    rng = np.random.default_rng(seed)
    requests, interarrivals = make_stream(rng)

    # Reference: the last L1 requests of the first segment.
    reference = normalized_counts(requests[SEG - L1:SEG], L1)

    sims = np.zeros(SIM_WINDOW)
    sim_means = []
    flags = []
    for i in range(SEG, len(requests)):
        window = requests[i - L2 + 1:i + 1]
        current = normalized_counts(window, L2)
        n = i - SEG + 1
        sims[(n - 1) % SIM_WINDOW] = cosine_similarity(reference, current)
        mean_sim = sims[:min(n, SIM_WINDOW)].mean()
        sim_means.append(mean_sim)
        if mean_sim < SIM_THRESHOLD:
            flags.append(i)

    # Rate detector over the whole stream.
    rate_flags = []
    buf = np.zeros(RATE_WINDOW)
    for i, x in enumerate(interarrivals):
        buf[i % RATE_WINDOW] = x
        if i >= RATE_WINDOW and abs(buf.mean() - RATE_MEAN) > RATE_TOL:
            rate_flags.append(i)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"cosine_detection_seed{seed}.npz")
    np.savez(out_path, similarity_mean=np.array(sim_means), popularity_flags=np.array(flags),
             rate_flags=np.array(rate_flags))
    first = flags[0] if flags else None
    print(f"popularity shift at request {SEG}; first cosine flag: {first}")
    print(f"rate flags: {len(rate_flags)}")
    print(f"saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    run(args.seed, args.out)
