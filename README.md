# Self-Pruning Neural Network on CIFAR-10

A feed-forward neural network that **learns to prune its own weights during training** using learnable sigmoid gates and L1 sparsity regularization — no post-training pruning step required.

Built for the Tredence AI Engineering Internship Case Study.

---

## How It Works

Each linear layer in the network has a second parameter tensor (`gate_scores`) with the same shape as the weight matrix. During the forward pass:

```
gates        = sigmoid(gate_scores)          # values in (0, 1)
pruned_weight = weight * gates               # element-wise mask
output        = pruned_weight @ x + bias
```

Training minimizes:
```
Total Loss = CrossEntropyLoss + λ * Σ(gates)
```

The L1 penalty on gates drives redundant connections to exactly zero, dynamically pruning the network while keeping the most important weights alive.

---

## Project Structure

```
self_pruning_nn/
├── src/
│   └── train.py          # PrunableLinear, model definition, training + evaluation loop
├── results/
│   ├── gate_hist_lambda_1.0.png
│   ├── gate_hist_lambda_5.0.png
│   ├── gate_hist_lambda_15.0.png
│   └── results.json
├── REPORT.md             # Analysis, results table, and gate distribution discussion
├── requirements.txt
└── README.md
```

---

## Setup & Run

### 1. Clone the repo

```bash
git clone https://github.com/tanishamanickavelan/self_pruning_nn
cd self_pruning_nn
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run training

```bash
python src/train.py
```

This will:
- Download CIFAR-10 automatically (first run only, ~170 MB)
- Train the network for 3 different λ values: `1.0`, `5.0`, `15.0`
- Print per-epoch accuracy and sparsity to the console
- Save gate histogram plots to `./results/`
- Save a summary JSON to `./results/results.json`

> **Note:** Training runs on CPU by default. With a CUDA GPU, it will be detected and used automatically. CPU training takes approximately 15–20 minutes per λ value (30 epochs).

---

## Results Summary

| Lambda (λ) | Test Accuracy (%) | Sparsity (%) |
|:----------:|:-----------------:|:------------:|
| 1.0        | 56.72             | 88.84        |
| 5.0        | **56.73**         | 98.67        |
| 15.0       | 56.70             | **99.86**    |

The network achieves ~57% accuracy on CIFAR-10 while pruning up to **99.86%** of its weights — with virtually no accuracy loss across all λ values tested.

See [`REPORT.md`](./REPORT.md) for the full analysis.

---

## Dependencies

- Python 3.8+
- PyTorch
- torchvision
- matplotlib