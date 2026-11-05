"""
Popularity change detection with KL divergence, plus a request-rate check.

The request stream has two segments. The first uses the Zipf popularity ranking. The second
uses the same Zipf distribution over a permuted set of file IDs, which is a popularity shift.
Arrivals are Poisson with mean interarrival time 0.2 throughout.

Popularity detector:
  - reference: normalized request counts over the first L1 = 1000 requests
  - current:   normalized request counts over the last L2 = 50 requests
  - divergence: KL(reference || current) = sum reference * log(reference / current)
  - flag: divergence > 8 (after 50 requests)

Rate detector (as in the notebook):
  - moving average of the last 10 interarrival times
  - flag: |average - 0.2| > 0.05

Usage:
    python change_detection_kld.py --seed 0 --out results
"""
import argparse
import os

import numpy as np

FILE_TYPES = 50
L1 = 1000        # reference window
L2 = 50          # current window
KLD_THRESHOLD = 8.0
EPS = 1e-10      # floor for zero counts in the current window
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


def kl_divergence(reference, current):
    """KL(reference || current). Terms with a zero reference count contribute nothing."""
    current = np.where(current == 0, EPS, current)
    mask = reference > 0
    return float(np.sum(reference[mask] * np.log(reference[mask] / current[mask])))


def run(seed, out_dir):
    rng = np.random.default_rng(seed)
    requests, interarrivals = make_stream(rng)

    # Reference: the last L1 requests of the first segment.
    reference = normalized_counts(requests[SEG - L1:SEG], L1)

    klds = []
    flags = []
    for i in range(SEG, len(requests)):
        window = requests[i - L2 + 1:i + 1]
        current = normalized_counts(window, L2)
        kld = kl_divergence(reference, current)
        klds.append(kld)
        if kld > KLD_THRESHOLD:
            flags.append(i)

    # Rate detector over the whole stream.
    rate_flags = []
    buf = np.zeros(RATE_WINDOW)
    for i, x in enumerate(interarrivals):
        buf[i % RATE_WINDOW] = x
        if i >= RATE_WINDOW and abs(buf.mean() - RATE_MEAN) > RATE_TOL:
            rate_flags.append(i)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"kld_detection_seed{seed}.npz")
    np.savez(out_path, kld=np.array(klds), popularity_flags=np.array(flags),
             rate_flags=np.array(rate_flags))
    first = flags[0] if flags else None
    print(f"popularity shift at request {SEG}; first KLD flag: {first}")
    print(f"rate flags: {len(rate_flags)}")
    print(f"saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    run(args.seed, args.out)
