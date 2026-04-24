# Self-Pruning Neural Network

This project is a small PyTorch implementation of a self-pruning MLP for CIFAR-10.
Instead of pruning after training, each weight has a learnable gate that is trained together with the model.
The idea is simple: important connections stay active, and weak ones get pushed down over time.

---

## Project Idea

Each dense layer uses a custom gated linear layer instead of `nn.Linear`.
During the forward pass, the gate scores go through a sigmoid and scale the weights:

```text
weight_final = weight * sigmoid(gate_score)
```

If a gate moves close to 1, the connection remains active.
If it moves close to 0, the connection becomes negligible and is effectively pruned.

---

## Model Setup

* Dataset: CIFAR-10
* Architecture: 3-layer MLP
* Activation: ReLU
* Layer type: custom gated linear layer

---

## Loss Function

The total loss is:

```text
Total Loss = CrossEntropy Loss + λ * Sparsity Loss
```

Where:

* CrossEntropy Loss handles classification
* Sparsity Loss is the sum of gate values (L1-style penalty)

This encourages the model to both learn the task and reduce unnecessary connections.

---

## Results

| Lambda | Accuracy | Sparsity |
| ------ | -------- | -------- |
| 0.01   | ~46%     | ~31%     |

> Note: Results may vary slightly depending on random seed and runtime.

---

## Observations

* Accuracy steadily improves during training
* Sparsity also increases over time, showing progressive pruning
* Increasing λ leads to higher sparsity but can reduce accuracy

This demonstrates the trade-off between model efficiency and performance.

---

## Why L1 Encourages Sparsity

The sparsity loss is based on the sum of gate values, which acts similar to L1 regularization.
L1-type penalties push parameters toward zero because smaller values reduce the loss.

Since gate values are passed through a sigmoid, pushing them toward zero reduces the contribution of corresponding weights, effectively pruning them.

---

## Gate Distribution Insight

The histogram of gate values (`out/gate_hist.png`) shows how weights are distributed after training.

* Values near 0 → pruned or inactive connections
* Values away from 0 → important retained weights

A noticeable concentration near zero indicates successful pruning.

---

## Output Files

* `out/model.pth` - trained model checkpoint
* `out/model.tar` - backup archive
* `out/gate_hist.png` - histogram of gate values

---

## How to Run

```bash
pip install -r requirements.txt
python main.py
```

The CIFAR-10 dataset downloads automatically on first run.

---

## Simple Explanation

The gate penalty pushes many gate values toward zero.
Since weights are multiplied by `sigmoid(gate_score)`, smaller gates reduce the impact of those weights.

As λ increases, the model focuses more on sparsity, which leads to more pruning.
However, if λ becomes too large, useful weights may also be removed, reducing accuracy.

---

## Notes

* Implementation is intentionally simple and easy to follow
* No external pruning libraries are used
* Focus is on learning pruning during training itself

