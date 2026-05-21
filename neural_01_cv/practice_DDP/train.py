import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


class SimpleMNISTNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(
        backend="gloo",
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )

    print(f"[rank {rank}] process started")

    dataset = MNIST(
        root="./data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=64,
        sampler=sampler,
        num_workers=0,
    )

    model = SimpleMNISTNet()
    model = DDP(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 3

    for epoch in range(epochs):
        sampler.set_epoch(epoch)

        epoch_loss = 0.0
        correct = 0
        total = 0

        start = time.time()

        for batch_idx, (x, y) in enumerate(loader):
            optimizer.zero_grad()

            pred = model(x)
            loss = loss_fn(pred, y)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            predicted_classes = pred.argmax(dim=1)
            correct += (predicted_classes == y).sum().item()
            total += y.size(0)

            if batch_idx % 100 == 0:
                print(
                    f"[rank {rank}] "
                    f"epoch={epoch} "
                    f"batch={batch_idx} "
                    f"loss={loss.item():.4f}"
                )

        local_loss = torch.tensor(epoch_loss / len(loader))
        local_correct = torch.tensor(correct)
        local_total = torch.tensor(total)

        dist.all_reduce(local_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_total, op=dist.ReduceOp.SUM)

        global_loss = local_loss / world_size
        global_accuracy = local_correct.float() / local_total.float()

        if rank == 0:
            print(
                f"\n[rank 0] epoch={epoch} "
                f"global_loss={global_loss.item():.4f} "
                f"global_accuracy={global_accuracy.item():.4f} "
                f"time={time.time() - start:.2f}s\n"
            )

    dist.barrier()

    if rank == 0:
        print("Training finished successfully.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()