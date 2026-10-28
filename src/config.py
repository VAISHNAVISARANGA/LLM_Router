"""Central place for the few constants shared across the project."""

# Models used as actions, ordered CHEAP -> EXPENSIVE.
# Action i == MODEL_POOL[i]. The last entry is treated as "the large model".
# These are column names inside RouterBench (see README for how to inspect them).
MODEL_POOL = ["mistralai/mistral-7b-chat",
            "mistralai/mixtral-8x7b-chat",
            "gpt-4-1106-preview",]

SEED = 42
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)  # train / val / test
