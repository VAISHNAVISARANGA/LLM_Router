"""Central place for the few constants shared across the project."""

# Models used as actions, ordered CHEAP -> EXPENSIVE.
# Action i == MODEL_POOL[i]. The last entry is treated as "the large model".
# These are column names inside RouterBench (see README for how to inspect them).
MODEL_POOL = [
    "liquid/lfm-2.5-2.6b:free"
    "nvidia/nemotron-3.5-lightning:free"
    "nvidia/nemotron-3-ultra-550b-a55b:free"
]

SEED = 42
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)  # train / val / test

# Balanced deployment setting chosen from validation quality/cost trade-offs.
DEFAULT_LAMBDA = 1.0
LAMBDA_SWEEP = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
