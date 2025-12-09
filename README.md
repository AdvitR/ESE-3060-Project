# ESE-3060 Project – CIFAR-10 Augmentation Experiments

This repository contains all experiment scripts and logging artifacts used for the CIFAR-10 Speedrun (Part 1, ESE-3060 Deep Learning). The goal of the project was to modify the `airbench94` training pipeline and evaluate how brightness/contrast jitter and curriculum augmentation schedules affect time-to-accuracy on CIFAR-10.

### **Experiment Scripts**

The following Python files implement the augmented training pipelines and automated sweeps:

* **`airbench_logging.py`** — Baseline `airbench94.py` with additional logging hooks for per-epoch metrics and benchmarking.
* **`airbench_bc_fixed.py`** — Version of airbench using **fixed-strength brightness/contrast jitter** added to the training loader.
* **`airbench_bc_curriculum.py`** — Version implementing a **curriculum augmentation schedule**, where jitter and translation strength increase over training.
* **`sweep_color_jitter.py`** — Script for running **hyperparameter sweeps** over brightness/contrast values (multiple seeds, automated logging).

These scripts allow full reproduction of all experiments reported in the paper.

---

### **Benchmarking & Training Logs**

All logs are stored as CSVs for easy analysis and plotting.

#### **50-seed result summaries**

* **`results_original.csv`** — Baseline airbench results.
* **`results_0.15_0.20.csv`** and suffixed variants

  * `_F`, `_M`, `_S`, `_3P` correspond to **four augmentation schedules** used in the fixed-jitter and curriculum experiments.
    These files contain **aggregate metrics across 50 random seeds**, including mean/variance of accuracy at epoch 8.9.

#### **Training-curve logs**

* **`training_curves_F.csv`**, **`training_curves_M.csv`**, **`training_curves_S.csv`**, **`training_curves_3P.csv`**
  These store **per-epoch training loss, training accuracy, validation accuracy, and timing curves** for each augmentation schedule.

#### **Hyperparameter sweep results**

* **`sweep_results_longer.csv`** — Results of the **extended brightness/contrast jitter sweep**, used to identify stable jitter ranges and evaluate performance sensitivity.
