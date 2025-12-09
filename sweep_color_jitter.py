import csv
import uuid
import os
from airbench_logging import *

# Hyperparameter Sweep Ranges
brightness_values = [0.00, 0.05, 0.10, 0.15, 0.20]
contrast_values   = [0.00, 0.05, 0.10, 0.15, 0.20]

# Number of runs to test per combination
SEEDS_PER_CONFIG = 3

# Output directory for sweep logs
log_dir = os.path.join("logs", f"sweep-{uuid.uuid4()}")
os.makedirs(log_dir, exist_ok=True)
csv_path = os.path.join(log_dir, "sweep_results.csv")

# Prepare CSV logging
fieldnames = [
    "brightness",
    "contrast",
    "seed",
    "time_to_target",
    "epoch_to_target",
    "final_acc",
    "total_time",
]

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    # Sweep Loop
    total_configs = len(brightness_values) * len(contrast_values)
    config_idx = 1

    for b in brightness_values:
        for c in contrast_values:

            print(f"\n=== Running Config {config_idx}/{total_configs} "
                  f"(brightness={b}, contrast={c}) ===")
            config_idx += 1

            # Set augmentation parameters
            hyp["aug"]["brightness"] = b
            hyp["aug"]["contrast"] = c

            # Run several seeds
            for seed in range(SEEDS_PER_CONFIG):
                print(f"  - Running seed {seed}")
                result = main(seed)

                # Add configuration details to results
                row = {
                    "brightness": b,
                    "contrast": c,
                    "seed": seed,
                    "time_to_target": result["time_to_target"],
                    "epoch_to_target": result["epoch_to_target"],
                    "final_acc": result["final_acc"],
                    "total_time": result["total_time"],
                }
                writer.writerow(row)

print("\nSweep completed!")
print("Saved sweep results to:", csv_path)
