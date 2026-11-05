# Transfer Learning with Prioritized PPO for Edge Caching

This repository contains the transfer-learning method for adaptive edge caching and a from-scratch baseline.

- `paper/Edge_Caching_Optimization_PPO_Transfer_Learning_IoTJ.pdf`: *Edge Caching Optimization with PPO and Transfer Learning for Dynamic Environments*, IEEE Internet of Things Journal, 2025. DOI: 10.1109/JIOT.2025.3577904.
- `code/TL_0_3_p.py`: the proposed method (two caches, three actions, critic transfer, demonstration transitions, reward-gap priorities).
- `code/1L_1_0.py`: the baseline that learns from scratch (single cache, two actions, no transfer).
- `code/file_types.csv`: lifetime, importance and size for each file type.
- `code/TL_0_3_p.ipynb` and `code/1L_1_0.ipynb`: the original notebooks, unchanged.

## Workload

Both scripts draw the same workload for a given seed and step count:
- **Requests:** file IDs 1–50 from a Zipf distribution with α = 1.
- **Arrivals:** a Poisson process with mean interarrival time 0.2 (exponential interarrivals, cumulative sum).
- **File attributes:** fixed per file type, read from `file_types.csv`.

## Running the transfer experiment

The proposed method needs a source run first to produce the critic weights and replay transitions.

```bash
pip install -r requirements.txt
cd code

# 1. Source run: saves critic.weights.h5 and demo.npz
python TL_0_3_p.py --steps 10000 --seed 0 --save-dir ../source

# 2. Target run: starts from the source critic and demonstrations
python TL_0_3_p.py --steps 10000 --seed 1 --source-dir ../source --out ../results

# Baseline: learns from scratch
python 1L_1_0.py --steps 10000 --seed 1 --out ../results
```

Each run writes an `.npz` file with per-step rewards, hit indicators after warm-up, and the hit ratio.

## Changes from the notebooks

- Per-request vectors for lifetime, importance, size, request stream and interarrivals were replaced with the generators above.
- Interarrivals are converted to arrival times with a cumulative sum. The notebooks used the interarrival values directly as absolute times.
- Cache-state presence and utility are indexed by file type. The notebooks indexed them by cache slot.
- The critic loss applies importance-sampling weights per sample.
- Returns and values have matching shapes.
- The HDF5 copy path was replaced by `demo.npz`.
- The JEQ demonstration margin term was omitted. For each sample it equals 0.8 × isdemo, which does not depend on the critic parameters, so it contributes no gradient.

## Transfer mechanism

- The source run's critic and replay buffer are loaded into the target run. The buffer entries are added as demonstrations (`isdemo = 1`) with their stored priorities.
- Priorities combine the absolute TD error with the reward gap |avg_R − r|, as in the notebook.
