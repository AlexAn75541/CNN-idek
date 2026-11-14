
# AGENT.md — EE300148 CPU-First CNN & Transfer Learning Project

## 1. Role & Objective
You are a senior systems and machine learning engineer building a **100% CPU-native, AVX2/AVX-512 optimized** Convolutional Neural Network (CNN) and Transfer Learning codebase for **Asia University / FPT Education Course EE300148 (Session 9 / Week 10)**.

Your mission is to translate the CUDA-oriented PyTorch transfer-learning pipeline from `Week10_CNNs_and_Colab.pptx` into a fast, deterministic, zero-cloud, CPU-first local repository targeting an **AMD Ryzen AI 9 (Zen 5 / Zen 5c, AVX2 + AVX-512)** processor, while preserving full structural and grading compliance with the course rubric.

## 2. Target Hardware & Runtime Constraints
- **Target CPU:** AMD Ryzen AI 9 (x86-64, AVX2 mandatory, AVX-512 VNNI/FP32 supported via oneDNN).
- **Forbidden Dependencies:** Zero `cuda` hardcoding (`torch.cuda.*` calls must never crash on CPU-only builds), zero NVIDIA GPU assumptions, zero Google Colab (`google.colab`) hard dependencies, zero cloud storage APIs, and zero manual `files.upload()` calls.
- **Backend Engine:** Standard PyTorch CPU (`torch` + `torchvision` CPU wheels) backed by Intel/AMD **oneDNN (MKL-DNN)** with `torch.channels_last` (NHWC) memory layout for AVX2/AVX-512 SIMD vectorization.
- **Deterministic Execution:** Fixed random seeds (`SEED = 42`) across `random`, `numpy`, and `torch`, plus controlled OpenMP thread pinning to physical cores to avoid SMT/hybrid-core cache thrashing.

## 3. Course Curriculum & Rubric Alignment (`Week10_CNNs_and_Colab.pptx`)
Every implementation must strictly satisfy the theoretical concepts and lab deliverables defined in the lecture slides:

### Part A — CNN Foundations & Transfer Learning Recipe (Slides 5–35)
1. **Manual Convolution Verification (Slides 9–17):**
   - Implement a pure NumPy/PyTorch reference check verifying output spatial dimensions:
     `output = floor((input - filter + 2 * padding) / stride) + 1`
   - Verify parameter savings of weight sharing (`Conv2d` vs. `Linear` dense layer on `32x32x3` inputs: `896` vs. `3,145,728` parameters).
2. **Data Preprocessing & Augmentation Pipeline (Slide 29):**
   - **Training Transform (`train_tf`):** `Resize((224, 224))`, `RandomHorizontalFlip()`, `RandomRotation(12)`, `ColorJitter(brightness=0.2)`, `ToTensor()`, `Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])`.
   - **Validation/Test Transform (`val_tf`):** `Resize((224, 224))`, `ToTensor()`, `Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])` — **NEVER** apply augmentation to validation or test splits.
3. **Two-Phase Transfer Learning Architecture (Slides 30–31):**
   - Backbone: `torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)` converted to `memory_format=torch.channels_last`.
   - **Phase 1 (Linear Probing / Frozen Backbone):** Freeze all backbone parameters (`p.requires_grad = False`), replace `model.fc = nn.Linear(model.fc.in_features, n_classes)`. Assert trainable parameter count is ~`2,565` (for `n_classes=5`), not `11.1M`. Train for `5 epochs` using `Adam(model.fc.parameters(), lr=1e-3)` and `CrossEntropyLoss()`.
   - **Phase 2 (Partial Unfreezing):** Unfreeze `model.layer4` (`p.requires_grad = True`). Train all active parameters for `5 epochs` using a 100x smaller learning rate: `Adam(..., lr=1e-5)`.
4. **Mandatory Lab Experiments & Diagnostics (Slide 32):**
   - Record validation accuracy after Phase 1 (5 epochs).
   - Record validation accuracy after Phase 2 (`layer4` unfrozen at `lr=1e-5`, 5 epochs) and log the delta.
   - **Ablation Study:** Disable training data augmentation (`train_tf = val_tf`), re-run Phase 1 from fresh pretrained weights, and record the validation accuracy drop.
   - **Error Analysis:** Compute the validation/test confusion matrix and programmatically identify the top two most confused classes.

### Part B — Reproducible Notebook & Report Structure (Slides 54–63)
The final deliverable (`notebook/EE300148_CNN_Project.ipynb` + modular `src/` package) must follow the exact **8-section structure** required for the Session 10 final report (40% of course grade):
1. **Setup:** CPU/AVX2/AVX-512 runtime diagnostics, thread pinning, package versions, deterministic seeds (`42`).
2. **Load:** Self-contained dataset fetcher (public URL or `torchvision.datasets` local cache) + immediate `.shape` and class count verification.
3. **Explore:** Dataset sample count ($n$), feature dimensions ($p$), missing/corrupt file check, class distribution bar chart, sample image grid.
4. **Split:** Stratified `train / validation / test` split with fixed seed **before** any fitting or augmentation.
5. **Baseline:** Trivial majority-class predictor (and/or frozen-feature Logistic Regression baseline) evaluated on the exact same split.
6. **Models:**
   - Model A: Custom lightweight CNN trained from scratch (demonstrating Slide 27's low-data regime gap).
   - Model B: ResNet-18 Phase 1 (frozen backbone + augmentation).
   - Model C: ResNet-18 Phase 2 (`layer4` unfrozen at `lr=1e-5`).
   - Ablation: ResNet-18 Phase 1 without augmentation.
   - Report `mean ± std` validation accuracy across folds or seeded runs.
7. **Final Evaluation:** Single-pass evaluation on the held-out `test` split (never touched during tuning).
8. **Figures & Conclusions:** Export high-DPI PNGs (`roc_curve.png`, `pr_curve.png`, `confusion_matrix.png`, `scratch_vs_transfer.png`), include a 1-sentence interpretation after every metric block, and draft the Limitations & AI-Use Declaration sections.

## 4. CPU SIMD (AVX2 / AVX-512) Coding Standards
1. **Thread Control:** At entry point (`src/cpu_runtime.py`), set `OMP_NUM_THREADS` and `MKL_NUM_THREADS` to physical core count (default `10`, configurable via CLI/env) before importing `torch` or `numpy`.
2. **Memory Layout:** Explicitly convert 4D image tensors and CNN modules using `.to(memory_format=torch.channels_last)` so oneDNN dispatches AVX2/AVX-512 blocked NCHWc/NHWC convolution kernels.
3. **Inference Mode:** Always wrap validation, test, and confusion matrix passes in `with torch.inference_mode():`.
4. **DataLoader Tuning:** On Windows CPU, set `num_workers=0` (or `2` with `persistent_workers=True`) and `pin_memory=False` (since `pin_memory=True` is CUDA-only and prints warnings on CPU).
5. **Checkpointing:** Save best validation checkpoints (`model.state_dict()`, `opt.state_dict()`, `epoch`, `val_acc`) to `./artifacts/checkpoints/best.pt` so training is resumable without cloud storage.
6. **Runtime Verification:** Print `torch.__config__.show()` filtered for `AVX2`, `AVX512`, and `MKL-DNN` in Section 1 so the marker/user can verify SIMD ISA acceleration is active.

## 5. Verification Gate (Definition of Done)
Before marking any task complete, the coding agent must verify:
- [ ] `python -m pytest tests/` passes (convolution shape math, `channels_last` tensor layout, deterministic seed invariance, baseline vs. transfer sanity check).
- [ ] Running `jupyter nbconvert --to notebook --execute notebook/EE300148_CNN_Project.ipynb` completes end-to-end from a clean restart on CPU in under 3 minutes with zero `CUDA` or `FileNotFoundError` exceptions.
- [ ] All four report figures and the Markdown summary table (`mean ± std`) are generated in `./artifacts/figures/` and `./artifacts/results.json`.