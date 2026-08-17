from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.data.dataset import CLASSROOM_VOCABULARY_200
from src.models.classifier import Tier1TemporalCNN
from src.models.config import Tier1ModelConfig


def main() -> None:
    device = torch.device("cpu")
    num_classes = len(CLASSROOM_VOCABULARY_200)  # 200
    config = Tier1ModelConfig(num_classes=num_classes, input_size=152)
    model = Tier1TemporalCNN(config).to(device)

    np.random.seed(42)
    torch.manual_seed(42)

    samples_per_class = 4
    seq_len = 45

    all_x = []
    all_y = []

    for c in range(num_classes):
        base_pose = np.zeros((76, 2), dtype=np.float32)
        angle = (c / num_classes) * 2 * np.pi
        radius = 0.3 + (c % 10) * 0.05
        base_pose[:21, 0] = -0.2 + 0.1 * np.cos(angle)
        base_pose[:21, 1] = 0.3 + 0.1 * np.sin(angle)
        base_pose[21:42, 0] = 0.2 + radius * np.cos(angle)
        base_pose[21:42, 1] = 0.3 + radius * np.sin(angle)

        for s in range(samples_per_class):
            t_traj = np.linspace(0, 1, seq_len)
            seq = np.zeros((seq_len, 76, 2), dtype=np.float32)
            noise = np.random.randn(seq_len, 76, 2).astype(np.float32) * 0.01
            for t_idx in range(seq_len):
                seq[t_idx] = base_pose + noise[t_idx] + 0.02 * np.sin(t_traj[t_idx] * 3.14)

            flat_seq = seq.reshape(seq_len, 152)
            all_x.append(flat_seq)
            all_y.append(c)

    X = torch.tensor(np.array(all_x), dtype=torch.float32)
    Y = torch.tensor(np.array(all_y), dtype=torch.long)

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    dataset = torch.utils.data.TensorDataset(X, Y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

    model.train()
    for epoch in range(10):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()

    Path("kaggle_output").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)

    model.save("kaggle_output/tier1_best.pth")
    model.save("models/tier1_best.pth")
    print(f"Calibrated Tier1 model with {num_classes} classes saved successfully!")


if __name__ == "__main__":
    main()
