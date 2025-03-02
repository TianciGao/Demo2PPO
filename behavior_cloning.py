"""
behavior_cloning.py

离线行为克隆(BC): 从 'demo_data.npz' 中加载(s,a)对, 用 MSE回归 a=pi(s).
输出 'bc_model.pth' 作为后续在线PPO微调的初始策略.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class BCPolicy(nn.Module):
    """
    与 SB3 MlpPolicy 类似的 2层64网络:
      obs_dim -> (Linear64 -> ReLU) -> (Linear64 -> ReLU) -> (Linear -> act_dim)
    """
    def __init__(self, obs_dim=12, act_dim=6, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, act_dim)
        )

    def forward(self, x):
        return self.net(x)


def train_bc_model(demo_file="demo_data.npz",
                   bc_model_file="bc_model.pth",
                   batch_size=64,
                   lr=1e-3,
                   max_epochs=20,
                   device="cpu"):
    """
    Args:
      demo_file (str): path to .npz file, storing an array of (s,a) pairs
      bc_model_file (str): output .pth for saving learned weights
      batch_size (int): training batch size
      lr (float): learning rate
      max_epochs (int): total epoch
      device (str): "cpu" or "cuda"

    Returns:
      bc_model (BCPolicy): the trained model
    """
    data = np.load(demo_file, allow_pickle=True)
    demos = data["demos"]  # (s, a) array
    print(f"[BC] Loaded {len(demos)} demonstration pairs from {demo_file}")

    # 分拆 (s, a)
    states, actions = [], []
    for (s_i, a_i) in demos:
        states.append(s_i)
        actions.append(a_i)
    states = np.array(states, dtype=np.float32)
    actions= np.array(actions, dtype=np.float32)

    print(f"[BC] states.shape={states.shape}, actions.shape={actions.shape}")
    if len(states)==0:
        print("[BC] No data => abort.")
        return None

    # 构建dataset
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(states),
        torch.from_numpy(actions)
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    obs_dim = states.shape[1]  # 12
    act_dim = actions.shape[1] # 6
    bc_model = BCPolicy(obs_dim, act_dim, hidden_size=64).to(device)

    optimizer = optim.Adam(bc_model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()  # or nn.SmoothL1Loss()

    print(f"[BC] device={device}, epochs={max_epochs}, batch_size={batch_size}, lr={lr}")
    # 训练循环
    for epoch in range(max_epochs):
        bc_model.train()
        epoch_loss=0.0
        for batch_states, batch_actions in dataloader:
            batch_states  = batch_states.to(device)
            batch_actions = batch_actions.to(device)

            pred_actions = bc_model(batch_states)
            loss = loss_fn(pred_actions, batch_actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss+= loss.item()
        epoch_loss/= len(dataloader)

        if (epoch+1)%5==0 or epoch==0:
            print(f"Epoch {epoch+1}/{max_epochs}, Loss={epoch_loss:.6f}")

    # 保存
    torch.save(bc_model.state_dict(), bc_model_file)
    print(f"[BC] Done training. Model saved to {bc_model_file}")
    return bc_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo_file", type=str, default="demo_data.npz", help="Path to demonstration data npz")
    parser.add_argument("--bc_model_file", type=str, default="bc_model.pth", help="Where to save BC model weights")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--use_cuda", action="store_true", help="Use CUDA if available")
    args= parser.parse_args()

    device = "cuda" if (args.use_cuda and torch.cuda.is_available()) else "cpu"
    train_bc_model(
        demo_file=args.demo_file,
        bc_model_file=args.bc_model_file,
        batch_size=args.batch_size,
        lr=args.lr,
        max_epochs=args.max_epochs,
        device=device
    )

if __name__=="__main__":
    main()
