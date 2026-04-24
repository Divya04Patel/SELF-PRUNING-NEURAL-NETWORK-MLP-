import argparse
import random
import tarfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader


seed = 42
bs = 128
lr = 1e-3
epochs = 8
img_dim = 32 * 32 * 3
hid = 512
num_cls = 10
data_root = Path("data")
out_dir = Path("out")
results_dir = Path("experiments")

out_dir.mkdir(parents=True, exist_ok=True)
results_dir.mkdir(parents=True, exist_ok=True)

torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class RunConfig:
    lambda_reg: float = 0.01
    epochs: int = epochs
    batch_size: int = bs
    learning_rate: float = lr
    seed: int = seed
    hidden_size: int = hid
    data_root: Path = data_root
    output_dir: Path = out_dir
    results_dir: Path = results_dir


class GatedLinear(nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out, inp) * 0.5)
        self.bias = nn.Parameter(torch.zeros(out))
        self.g = nn.Parameter(torch.randn(out, inp) * 0.7 - 4.0)

    def forward(self, x):
        gate = torch.sigmoid(self.g)
        w = self.weight * gate
        return F.linear(x, w, self.bias)

    def sp_loss(self):
        return torch.sigmoid(self.g).sum()

    def effective_weight(self):
        return self.weight * torch.sigmoid(self.g)

    def gate_vals(self):
        return torch.sigmoid(self.g).detach().flatten().cpu()


class Net(nn.Module):
    def __init__(self, hidden_size=hid):
        super().__init__()
        self.fc1 = GatedLinear(img_dim, hidden_size)
        self.fc2 = GatedLinear(hidden_size, hidden_size)
        self.fc3 = GatedLinear(hidden_size, num_cls)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sp_loss(self):
        return self.fc1.sp_loss() + self.fc2.sp_loss() + self.fc3.sp_loss()

    def prunable_weights(self):
        return [self.fc1.effective_weight(), self.fc2.effective_weight(), self.fc3.effective_weight()]

    def all_gates(self):
        return torch.cat([self.fc1.gate_vals(), self.fc2.gate_vals(), self.fc3.gate_vals()])


def set_seed(run_seed):
    random.seed(run_seed)
    torch.manual_seed(run_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run_seed)


def get_data(batch_size, root):
    train_transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    train_ds = torchvision.datasets.CIFAR10(root=str(root), train=True, download=True, transform=train_transform)
    test_ds = torchvision.datasets.CIFAR10(root=str(root), train=False, download=True, transform=test_transform)

    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_ld = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_ld, test_ld


def compute_sparsity(model, threshold=1e-3):
    total_weights = 0
    near_zero_weights = 0

    for weight_tensor in model.prunable_weights():
        total_weights += weight_tensor.numel()
        near_zero_weights += (weight_tensor.abs() < threshold).sum().item()

    sparsity_percent = 100.0 * near_zero_weights / total_weights if total_weights else 0.0
    return total_weights, near_zero_weights, sparsity_percent


def train_one_epoch(net, train_ld, opt, lambda_reg_value):
    net.train()
    total_loss = 0.0

    for x, y in train_ld:
        x = x.to(device)
        y = y.to(device)

        opt.zero_grad()
        out = net(x)
        original_loss = F.cross_entropy(out, y)
        regularization_term = net.sp_loss()
        loss = original_loss + lambda_reg_value * regularization_term
        loss.backward()
        opt.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(train_ld.dataset)


@torch.no_grad()
def test(net, test_ld):
    net.eval()
    correct = 0
    total = 0

    for x, y in test_ld:
        x = x.to(device)
        y = y.to(device)
        out = net(x)
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return correct / total


def plot_gates(net, output_dir):
    g = net.all_gates().numpy()
    plt.figure(figsize=(7, 4))
    plt.hist(g, bins=40, color="steelblue", edgecolor="black")
    plt.title("final gate values")
    plt.xlabel("sigmoid(g)")
    plt.ylabel("count")
    plt.tight_layout()
    fp = output_dir / "gate_hist.png"
    plt.savefig(fp, dpi=150)
    plt.close()
    return fp


def format_lambda(lambda_reg_value):
    return f"{lambda_reg_value:g}"


def save_run_results(config, final_acc, sparsity_percent, total_weights, near_zero_weights):
    results_path = config.results_dir / f"run_{format_lambda(config.lambda_reg)}.txt"
    with results_path.open("w", encoding="utf-8") as f:
        f.write(f"lambda_reg: {config.lambda_reg}\n")
        f.write(f"final_test_accuracy: {final_acc:.6f}\n")
        f.write(f"sparsity_percent: {sparsity_percent:.4f}\n")
        f.write(f"total_weights: {total_weights}\n")
        f.write(f"near_zero_weights: {near_zero_weights}\n")
    return results_path


def run_training(config):
    set_seed(config.seed)

    train_ld, test_ld = get_data(config.batch_size, config.data_root)
    net = Net(hidden_size=config.hidden_size).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=config.learning_rate)
    run_output_dir = config.output_dir / f"lambda_{format_lambda(config.lambda_reg)}"
    run_output_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(1, config.epochs + 1):
        tr_loss = train_one_epoch(net, train_ld, opt, config.lambda_reg)
        acc = test(net, test_ld)
        total_weights, near_zero_weights, sparsity_percent = compute_sparsity(net)
        print(
            f"epoch {ep:02d} | loss {tr_loss:.4f} | test acc {acc*100:.2f}% | sparsity {sparsity_percent:.2f}%"
        )

    final_acc = test(net, test_ld)
    total_weights, near_zero_weights, sparsity_percent = compute_sparsity(net)
    gate_plot_path = plot_gates(net, run_output_dir)

    pth_fp = run_output_dir / "model.pth"
    tar_fp = run_output_dir / "model.tar"

    buf = BytesIO()
    torch.save(net.state_dict(), buf)
    buf.seek(0)
    torch.save(net.state_dict(), pth_fp)
    with tarfile.open(tar_fp, "w") as tar:
        info = tarfile.TarInfo(name="model_state_dict.pth")
        info.size = len(buf.getbuffer())
        tar.addfile(info, buf)

    results_path = save_run_results(config, final_acc, sparsity_percent, total_weights, near_zero_weights)

    print(f"lambda_reg: {config.lambda_reg}")
    print(f"final test accuracy: {final_acc * 100:.2f}%")
    print(f"sparsity: {sparsity_percent:.2f}%")
    print("gate plot saved to:", gate_plot_path)
    print("model saved to:", pth_fp)
    print("tar backup saved to:", tar_fp)
    print("run results saved to:", results_path)

    return {
        "lambda_reg": config.lambda_reg,
        "final_test_accuracy": final_acc,
        "sparsity_percent": sparsity_percent,
        "total_weights": total_weights,
        "near_zero_weights": near_zero_weights,
        "results_path": results_path,
    }


def build_parser():
    parser = argparse.ArgumentParser(description="Train a self-pruning MLP with learnable gates")
    parser.add_argument("--lambda_reg", type=float, default=0.01, help="Regularization strength for gate sparsity")
    parser.add_argument("--epochs", type=int, default=epochs, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=bs, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=lr, help="Learning rate")
    parser.add_argument("--seed", type=int, default=seed, help="Random seed")
    parser.add_argument("--hidden_size", type=int, default=hid, help="Hidden layer width")
    parser.add_argument("--data_root", type=Path, default=data_root, help="Dataset root directory")
    parser.add_argument("--output_dir", type=Path, default=out_dir, help="Directory for plots and model files")
    parser.add_argument("--results_dir", type=Path, default=results_dir, help="Directory for experiment result text files")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    config = RunConfig(
        lambda_reg=args.lambda_reg,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
        hidden_size=args.hidden_size,
        data_root=args.data_root,
        output_dir=args.output_dir,
        results_dir=args.results_dir,
    )
    print("device:", device)
    run_training(config)


if __name__ == "__main__":
    main()
