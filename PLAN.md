# PLAN.md — Execution Roadmap for CPU-Native CNN & Transfer Learning (EE300148)

## Overview
Build a modular, self-contained Python repository and reproducible Jupyter Notebook implementing the full **EE300148 Week 10 CNN & Transfer Learning curriculum** exclusively on local x86-64 CPU hardware with **AVX2 / AVX-512 (oneDNN)** acceleration.

---

## Repository Directory Structure
```text
ee300148-cpu-cnn/
├── AGENT.md                          # Agent behavioral & architectural specification
├── PLAN.md                           # Step-by-step execution roadmap
├── pyproject.toml                    # Dependencies (CPU PyTorch, torchvision, ISLP, scikit-learn, matplotlib)
├── src/
│   ├── __init__.py
│   ├── cpu_runtime.py                # AVX2/AVX-512 detection, OMP/MKL thread pinning, seed setup
│   ├── cnn_math.py                   # Hand-worked 2D conv verifier & parameter-count calculator (Slides 9-20)
│   ├── dataset.py                    # Self-contained dataset loader, stratified split, ImageNet transforms
│   ├── models.py                     # Baseline classifier, ScratchCNN, and channels_last ResNet-18 factory
│   ├── engine.py                     # CPU-optimized train/eval loops, Phase 1/2 scheduler, checkpointing
│   └── metrics.py                    # Confusion matrix, ROC/PR curves, top-2 confused classes analyzer
├── notebook/
│   └── EE300148_CNN_Project.ipynb    # 8-section reproducible report notebook (Slide 54 compliant)
├── tests/
│   ├── test_cnn_math.py              # Unit tests for Slide 10 output formula & Slide 20 parameter counts
│   └── test_pipeline_cpu.py          # Smoke tests for channels_last, freezing, unfreezing, and CPU inference
└── artifacts/
    ├── checkpoints/                  # Local .pt checkpoints (Slide 50)
    └── figures/                      # Exported ROC, PR, Confusion Matrix, and Scratch-vs-Transfer PNGs
```

---

## Phase-by-Phase Implementation Plan

### Phase 1: Environment & CPU SIMD Runtime (`src/cpu_runtime.py`)
1. **Configure Threading & SIMD:**
   - Set `os.environ["OMP_NUM_THREADS"]`, `os.environ["MKL_NUM_THREADS"]`, and `torch.set_num_threads()` to physical core count (default `10` for Ryzen AI 9, auto-detectable via `psutil.cpu_count(logical=False)`).
   - Enable `torch.backends.mkldnn.enabled = True`.
   - Implement `inspect_cpu_simd()` to parse `torch.__config__.show()` and confirm `AVX2`, `AVX512`, and `oneDNN/MKL-DNN` support.
2. **Deterministic Seeding:**
   - Implement `seed_everything(seed=42)` setting `PYTHONHASHSEED`, `random.seed`, `np.random.seed`, and `torch.manual_seed`.

### Phase 2: CNN Theory & Math Verification Module (`src/cnn_math.py`)
1. **Convolution Output Formula (Slide 10 & Slide 17):**
   - Implement `conv2d_output_size(input_size, kernel_size, padding=0, stride=1)` and validate against `torch.nn.Conv2d` on the Slide 16 checkpoint (`32x32` input, `5x5` kernel, `padding=0`, `stride=1` -> `28x28`).
2. **Hand-Worked Vertical Edge Kernel Demo (Slide 9):**
   - Implement `verify_hand_convolution()` reproducing the exact `5x5` input matrix and `3x3` vertical edge filter (`+1, 0, -1`) from Slide 9, asserting the `3x3` feature map matches `[[-4, 4, -1], [-6, 1, 3], [-3, -4, 5]]`.
3. **Parameter Efficiency Calculator (Slide 20):**
   - Implement `compare_dense_vs_conv_params()` proving a `32x32x3` input mapped to `1,024` dense units uses `3,145,728` weights vs. `896` weights for `32` filters of `3x3x3` (`~3,510x` reduction).

### Phase 3: Self-Contained Dataset & Stratified Splits (`src/dataset.py`)
1. **Zero-Account Dataset Acquisition:**
   - Default dataset: `torchvision.datasets.Flowers102` filtered to `n_classes = 5` (matching Slide 30 `n_classes = 5` and Slide 32 "provided flower dataset"), cached locally in `./data/flowers102`.
   - Support drop-in custom `ImageFolder` path via `--data-dir` so custom image folders work without code changes.
2. **Strict Train / Val / Test Split (Slide 54 Step 4):**
   - Perform stratified splitting (`60% train / 20% val / 20% test`) using `SEED = 42` prior to applying transforms.
3. **Transform Pipelines (Slide 29):**
   - `get_transforms(augment=True)` returning `(train_tf, val_tf)` with exact ImageNet normalization (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).
   - `get_transforms(augment=False)` for the Slide 32 augmentation ablation experiment.

### Phase 4: Models & CPU `channels_last` Optimization (`src/models.py`)
1. **Baseline Predictor (Slide 54 Step 5):**
   - `MajorityClassBaseline` predicting the most frequent training class to establish the floor accuracy, ROC-AUC (`0.50`), and PR-AUC.
2. **Small CNN From Scratch (Slide 22 & Slide 27):**
   - `ScratchCNN(n_classes)`: 2 blocks of `Conv2d -> ReLU -> MaxPool2d` (`32` and `64` filters) + `Flatten -> Linear(128) -> ReLU -> Dropout(0.3) -> Linear(n_classes)`, converted to `torch.channels_last`. Demonstrates why training from scratch underperforms on a few hundred images.
3. **Pretrained ResNet-18 Transfer Model (Slide 30):**
   - `build_resnet18_transfer(n_classes)`: Loads `ResNet18_Weights.DEFAULT`, sets `requires_grad = False` for all parameters, replaces `model.fc = nn.Linear(512, n_classes)`, converts to `memory_format=torch.channels_last`, and asserts `count_trainable_params(model) == 512 * n_classes + n_classes`.
   - `unfreeze_layer4(model)`: Sets `requires_grad = True` exclusively on `model.layer4` parameters.

### Phase 5: Training Engine, Ablation & Checkpointing (`src/engine.py`)
1. **CPU Training Loop:**
   - Convert input batches via `x = x.to(memory_format=torch.channels_last)`.
   - Use `torch.inference_mode()` during validation/testing.
   - Save best model + optimizer state dict to `./artifacts/checkpoints/` whenever `val_acc` improves (Slide 50).
2. **Experiment Runner (`run_all_experiments`):**
   - Executes across 3 seeds (`[42, 100, 2026]`) to compute `mean ± std` (required by Slide 62):
     1. Majority-Class Baseline
     2. `ScratchCNN` (10 epochs, `lr=1e-3`)
     3. `ResNet-18` Phase 1 Frozen Head + Augmentation (5 epochs, `lr=1e-3`)
     4. `ResNet-18` Phase 2 Unfreeze `layer4` + Augmentation (5 epochs, `lr=1e-5`)
     5. `ResNet-18` Phase 1 Frozen Head **Without** Augmentation (5 epochs, `lr=1e-3`)

### Phase 6: Evaluation Metrics & Report Figures (`src/metrics.py`)
1. **Held-Out Test Evaluation (Slide 54 Step 7):**
   - Load best checkpoint and evaluate **once** on `test_loader`.
2. **Diagnostic Plots (Slide 32 & Slide 62):**
   - `confusion_matrix.png`: Multi-class confusion matrix + automated printout of the **top 2 most confused class pairs**.
   - `roc_pr_curves.png`: One-vs-Rest ROC and Precision-Recall curves with baseline reference lines.
   - `scratch_vs_transfer.png`: Bar chart comparing `mean ± std` test accuracy across Baseline, Scratch CNN, Transfer Phase 1 (No Aug), Transfer Phase 1 (Aug), and Transfer Phase 2 (`layer4`).

### Phase 7: Reproducible 8-Section Jupyter Notebook (`notebook/EE300148_CNN_Project.ipynb`)
1. Assemble the complete notebook following Slide 54's exact 8 sections:
   - `1. Setup` -> `2. Load` -> `3. Explore` -> `4. Split` -> `5. Baseline` -> `6. Models` -> `7. Final Evaluation` -> `8. Figures & Conclusions`.
2. Ensure every code section is preceded by a concise Markdown explanation and every metric output is followed by a 1-sentence analytical interpretation (Slide 55).
3. Include drafted **Limitations** and **AI-Use Declaration** sections (Slide 62).
4. Execute full verification via `nbconvert --execute` to guarantee zero errors on clean restart.