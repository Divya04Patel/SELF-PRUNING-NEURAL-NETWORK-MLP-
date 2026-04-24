# Self-Pruning Neural Network

This project is a small PyTorch implementation of a self-pruning MLP for CIFAR-10.
Instead of pruning after training, each weight has a learnable gate that is trained together with the model.
The idea is simple: important connections stay active, and weak ones get pushed down over time.

## Project Idea

Each dense layer uses a custom gated linear layer instead of `nn.Linear`.
During the forward pass, the gate scores go through a sigmoid and scale the weights:

```text
weight_final = weight * sigmoid(gate_score)
```

So if a gate moves close to 1, that connection stays active.
If it moves close to 0, the connection becomes very small and is effectively pruned.

## Model Setup

- Dataset: CIFAR-10
- Architecture: 3-layer MLP
- Activation: ReLU
- Layer type: custom gated linear layer

## Loss Function

The total loss is:

```text
Total Loss = CrossEntropy Loss + λ * Sparsity Loss
```

Where:

- CrossEntropy Loss is for classification
- Sparsity Loss is the sum of gate values across layers

This makes the model learn classification and pruning at the same time.

## Results

Example short-run results from this version:

| Metric | Value |
| --- | --- |
| Accuracy | ~46% |
| Sparsity | ~31% |

Results can change depending on the machine, random seed, and training time.

## Output Files

- `out/model.pth` - main trained checkpoint
- `out/model.tar` - backup archive containing the saved state dict
- `out/gate_hist.png` - histogram of final gate values

## How to Run

```bash
pip install -r requirements.txt
python main.py
```

The CIFAR-10 dataset downloads automatically the first time.

## Simple Explanation

The gate penalty pushes gate values toward zero during training.
Since every weight is multiplied by `sigmoid(gate_score)`, smaller gates make the corresponding weights weaker.
That is why the model starts removing less useful connections on its own.

When `λ` increases, the model focuses more on sparsity.
That usually gives more pruning, but if it becomes too strong, accuracy can drop because the model loses useful weights.

## Notes

- The code is intentionally kept simple and easy to follow
- No pruning libraries or advanced training framework are used
- The goal is to show self-pruning learned directly during training
