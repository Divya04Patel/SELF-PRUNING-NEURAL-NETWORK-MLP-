import warnings
import random
import tarfile
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader


warnings.filterwarnings("ignore", message=r"dtype\(\): align should be passed.*")


seed = 42
bs = 128
lr = 1e-3
epochs = 8
lam = 0.01
img_dim = 32 * 32 * 3
hid = 512
num_cls = 10
data_root = Path("data")
out_dir = Path("out")
out_dir.mkdir(parents=True, exist_ok=True)

random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    def gate_vals(self):
        return torch.sigmoid(self.g).detach().flatten().cpu()

    def sparsity(self):
        gate = torch.sigmoid(self.g)
        return (gate < 1e-2).float().mean().item()


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = GatedLinear(img_dim, hid)
        self.fc2 = GatedLinear(hid, hid)
        self.fc3 = GatedLinear(hid, num_cls)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sp_loss(self):
        return self.fc1.sp_loss() + self.fc2.sp_loss() + self.fc3.sp_loss()

    def sparsity(self):
        g = torch.cat([
            torch.sigmoid(self.fc1.g).flatten(),
            torch.sigmoid(self.fc2.g).flatten(),
            torch.sigmoid(self.fc3.g).flatten(),
        ])
        return (g < 1e-2).float().mean().item()

    def all_gates(self):
        return torch.cat([self.fc1.gate_vals(), self.fc2.gate_vals(), self.fc3.gate_vals()])


def get_data():
    tr = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    train_ds = torchvision.datasets.CIFAR10(root=str(data_root), train=True, download=True, transform=tr)
    test_ds = torchvision.datasets.CIFAR10(root=str(data_root), train=False, download=True, transform=tr)

    train_ld = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0)
    test_ld = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=0)
    return train_ld, test_ld


def train(net, train_ld, opt):
    net.train()
    total = 0.0
    for x, y in train_ld:
        x = x.to(device)
        y = y.to(device)
        opt.zero_grad()
        out = net(x)
        ce = F.cross_entropy(out, y)
        sp_loss = net.sp_loss()
        loss = ce + lam * sp_loss
        loss.backward()
        opt.step()
        total += loss.item() * x.size(0)
    return total / len(train_ld.dataset)


@torch.no_grad()
def test(net, test_ld):
    net.eval()
    cor = 0
    tot = 0
    for x, y in test_ld:
        x = x.to(device)
        y = y.to(device)
        out = net(x)
        pred = out.argmax(1)
        cor += (pred == y).sum().item()
        tot += y.size(0)
    return cor / tot


def plot_gates(net):
    g = net.all_gates().numpy()
    plt.figure(figsize=(7, 4))
    plt.hist(g, bins=40, color="steelblue", edgecolor="black")
    plt.title("final gate values")
    plt.xlabel("sigmoid(g)")
    plt.ylabel("count")
    plt.tight_layout()
    fp = out_dir / "gate_hist.png"
    plt.savefig(fp, dpi=150)
    plt.close()
    return fp


def main():
    print("device:", device)
    train_ld, test_ld = get_data()
    net = Net().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    for ep in range(1, epochs + 1):
        tr_loss = train(net, train_ld, opt)
        acc = test(net, test_ld)
        sp = net.sparsity() * 100.0
        print(f"epoch {ep:02d} | loss {tr_loss:.4f} | test acc {acc*100:.2f}% | sparsity {sp:.2f}%")

    acc = test(net, test_ld)
    sp = net.sparsity() * 100.0
    fp = plot_gates(net)
    print("final acc:", f"{acc*100:.2f}%")
    print("final sparsity:", f"{sp:.2f}%")
    print("gate plot saved to:", fp)

    buf = BytesIO()
    torch.save(net.state_dict(), buf)
    buf.seek(0)
    pth_fp = out_dir / "model.pth"
    torch.save(net.state_dict(), pth_fp)
    tar_fp = out_dir / "model.tar"
    with tarfile.open(tar_fp, "w") as tar:
        info = tarfile.TarInfo(name="model_state_dict.pth")
        info.size = len(buf.getbuffer())
        tar.addfile(info, buf)
    print("model saved to:", pth_fp)
    print("tar backup saved to:", tar_fp)

    print()
    print("plain explanation:")
    print("L1-ish penalty on gates keeps pushing them down, so sigmoid(g) gets smaller and smaller.")
    print("When lambda is bigger, the model cares more about sparsity, so more weights get shut down.")
    print("That usually gives more pruning but can hurt accuracy if lambda is too strong.")


if __name__ == "__main__":
    main()
