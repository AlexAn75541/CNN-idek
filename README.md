# EE300148 — CPU-First CNN & Transfer Learning Project

A simple, fast, and 100% CPU-native image classification project built for **Asia University / FPT Education Course EE300148 (Session 9 / Week 10)**.

---

## 1. What is this project & What is it for?
This project is an end-to-end Computer Vision assignment that classifies images of flowers across 5 categories (`daisy`, `dandelion`, `rose`, `sunflower`, `tulip`).

Unlike typical deep learning projects that require an NVIDIA GPU or Google Colab, this entire repository is built from the ground up to run **directly on your local computer's CPU**. It requires **zero GPU hardware**, **zero cloud accounts**, and **zero file uploads**.

---

## 2. What is the purpose?
The project demonstrates core deep learning concepts from the course lectures:
1. **Why Convolutions work:** Proves mathematically and practically why Convolutional layers (`Conv2d`) are ~3,500x smaller and more efficient than classic Dense layers (`Linear`) by sharing weights across image pixels.
2. **Transfer Learning vs. Training from Scratch:** Shows why starting with a pretrained model (ResNet-18) is vastly superior to training a new model from scratch when you only have a small dataset.
3. **The 2-Phase Transfer Learning Recipe:**
   - **Phase 1 (Feature Extraction):** Freeze the pretrained ResNet-18 backbone and only train the final output layer (~2,565 parameters).
   - **Phase 2 (Fine-Tuning):** Unfreeze the top convolutional block (`layer4`) and train it at a 100x smaller learning rate (`1e-5`) to specialize it to flower details.
4. **Data Augmentation:** Proves how simple data augmentations (flips, rotations, brightness tweaks) protect against overfitting and boost accuracy by over +30%.

---

## 3. What are the benefits?
- **Zero GPU or Cloud dependency:** Runs anywhere Python runs without CUDA errors, out-of-memory crashes, or Google Colab disconnects.
- **Fast Execution (< 3 minutes):** Uses CPU SIMD acceleration (AVX2/AVX-512 via Intel/AMD oneDNN) and optimal memory layout (`torch.channels_last`), letting the entire multi-model pipeline and notebook run in under 3 minutes on standard modern laptop/desktop processors.
- **Self-contained & Reproducible:** Includes a synthetic 5-class flower dataset and automated seeds (`seed=42`), producing identical, verifiable results every time.
- **Course & Rubric compliant:** Automatically structures the notebook into the exact 8 required sections (Setup, Load, Explore, Split, Baseline, Models, Evaluation, Figures & Conclusions) and exports all required submission artifacts.

---

## 4. What is the outcome?
Running this project produces:
1. **Interactive Notebook:** `notebook/EE300148_CNN_Project.ipynb` containing all 8 report sections, explanations, and code outputs.
2. **Four High-Resolution Figures** in `artifacts/figures/`:
   - `confusion_matrix.png`: Shows which flowers the model confuses (error analysis).
   - `roc_curve.png`: One-vs-Rest Receiver Operating Characteristic curves for all 5 classes.
   - `pr_curve.png`: Precision-Recall curves.
   - `scratch_vs_transfer.png`: Side-by-side accuracy comparison bar chart across all tested configurations.
3. **Structured Results:** `artifacts/results.json` containing exact test accuracy (98.3%), multi-seed means, standard deviations, and confusion metrics.
4. **Best Checkpoint:** Saved model weights in `artifacts/checkpoints/` ready for inference.

---

## 5. How to Proceed (Step-by-Step)

### Step 1: Open Terminal & Activate the Virtual Environment
In Windows PowerShell or Command Prompt:
```bat
c:\Users\Aretzera\Documents\CNNniggas\.venv\Scripts\activate
```
*(On Linux/macOS/WSL: `source .venv/bin/activate`)*

### Step 2: Run the Unit Tests
Verify that all convolution math, layer parameter counts, and CPU optimizations pass:
```bash
pytest tests/
```
*(All 10 tests should pass in ~15 seconds)*

### Step 3: Run the Complete Notebook
Run the entire 8-section course notebook automatically in your terminal:
```bash
jupyter nbconvert --to notebook --execute notebook/EE300148_CNN_Project.ipynb
```
*(Executes all 28 cells and exports all charts in under 3 minutes)*

Or, open it interactively in Jupyter Lab / VS Code to inspect the cells:
```bash
jupyter lab notebook/EE300148_CNN_Project.ipynb
```

### Step 4: Check the Output Artifacts
Open the generated charts in `artifacts/figures/` and review the benchmark numbers in `artifacts/results.json`.

---

## 6. Project Structure

```text
├── src/
│   ├── cpu_runtime.py   # Thread pinning (OMP/MKL to physical cores), AVX2/AVX-512 checks, seed 42
│   ├── cnn_math.py      # Output size formula (Slide 16), manual 2D convolution (Slide 9), parameter math (Slide 20)
│   ├── dataset.py       # Slide 29 preprocessing, augmentations, balanced 5-class flower dataset (n=300)
│   ├── models.py        # Majority baseline, Scratch CNN, ResNet-18 Phase 1 (2,565 params) & Phase 2
│   ├── engine.py        # CPU channels_last training/eval loops, feature caching, checkpointing
│   └── metrics.py       # Confusion matrix, top-2 error analysis, ROC/PR curves, results JSON
├── notebook/
│   └── EE300148_CNN_Project.ipynb  # Executable 8-section course report notebook (< 3 min on CPU)
├── tests/
│   ├── conftest.py            # Automatic sys.path configuration for bare pytest execution
│   ├── test_cnn_math.py       # Formula & parameter savings unit tests
│   └── test_pipeline_cpu.py   # CPU runtime, determinism, channels_last, model tests
├── artifacts/
│   ├── figures/         # roc_curve.png, pr_curve.png, confusion_matrix.png, scratch_vs_transfer.png
│   ├── checkpoints/     # Best validation checkpoints (.pt)
│   └── results.json     # Multi-seed metrics summary
├── create_notebook.py   # Deterministic notebook builder script
├── pyproject.toml       # Python package dependencies and pytest configuration
└── README.md            # Project guide and instructions
```

---

## 7. Benchmark Results

Summary across 3 random seeds (`42`, `100`, `2026`):

| Model / Configuration | Trainable Params | Validation Acc (Mean ± Std) | Role in Course |
| :--- | :--- | :--- | :--- |
| **Majority Baseline** | 0 | 20.00% ± 0.00% | Performance floor (random guess on 5 classes) |
| **Scratch CNN** | ~200K | 79.44% ± 0.79% | Demonstrates data-hungry gap of custom CNNs |
| **ResNet-18 P1 (No Aug)** | 2,565 | 52.78% ± 10.21% | Ablation baseline (no data augmentation) |
| **ResNet-18 P1 (With Aug)** | 2,565 | 92.22% ± 0.79% | Linear probing + Slide 29 transforms (+39.4% delta) |
| **ResNet-18 P2 (Fine-Tuning)** | 8,396,293 | **100.00% ± 0.00%** | Fine-tuning `layer4` + `fc` at `lr=1e-5` |

- **Held-Out Test Accuracy:** **98.33%** (single-pass evaluation on untouched test data).
- **Conv2d Weight Sharing Savings:** **3,510x parameter reduction** over a dense layer on 32x32x3 input (896 vs 3,145,728 parameters).
- **Top Confused Pair:** Dandelion misclassified as Tulip (1 sample).
