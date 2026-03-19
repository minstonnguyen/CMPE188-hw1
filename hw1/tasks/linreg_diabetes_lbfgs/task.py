import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_task_metadata():
    return {
        "task_id": "linreg_diabetes_lbfgs",
        "description": "Linear regression with L-BFGS (quasi-Newton) optimizer",
        "input_dim": 10,
        "output_dim": 1,
    }


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_dataloaders(train_ratio=0.8, batch_size=200):
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    data = load_diabetes()
    X = data.data.astype(np.float32)
    y = data.target.astype(np.float32).reshape(-1, 1)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, train_size=train_ratio, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    train_loader = DataLoader(
        train_ds, batch_size=max(batch_size, len(X_train)), shuffle=False
    )
    val_loader = DataLoader(val_ds, batch_size=len(X_val), shuffle=False)
    return train_loader, val_loader


def build_model(device=None):
    device = device or get_device()
    return nn.Sequential(nn.Linear(10, 1)).to(device)


def train(model, train_loader, val_loader, device, max_iter=50):
    model.train()
    batch = next(iter(train_loader))
    X_all = batch[0].to(device)
    y_all = batch[1].to(device)
    optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=20)

    def closure():
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(model(X_all), y_all)
        loss.backward()
        return loss

    for i in range(max_iter):
        optimizer.step(closure)
        if (i + 1) % 10 == 0:
            with torch.no_grad():
                cur_loss = nn.functional.mse_loss(model(X_all), y_all).item()
            print(f"Iter {i+1}/{max_iter}, Loss: {cur_loss:.6f}")


def evaluate(model, data_loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in data_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds.append(model(xb).cpu())
            targets.append(yb.cpu())
    y_pred = torch.cat(preds)
    y_true = torch.cat(targets)
    mse = torch.mean((y_pred - y_true) ** 2).item()
    ss_res = torch.sum((y_true - y_pred) ** 2).item()
    ss_tot = torch.sum((y_true - y_true.mean()) ** 2).item()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"mse": mse, "r2": r2}


def predict(model, X, device):
    model.eval()
    if not isinstance(X, torch.Tensor):
        X = torch.FloatTensor(X)
    with torch.no_grad():
        return model(X.to(device)).cpu().numpy()


def save_artifacts(model, metrics, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))


if __name__ == "__main__":
    set_seed(42)
    device = get_device()
    print("Task: Linear Regression (L-BFGS)")

    train_loader, val_loader = make_dataloaders()
    model = build_model(device=device)
    train(model, train_loader, val_loader, device, max_iter=50)

    train_m = evaluate(model, train_loader, device)
    val_m = evaluate(model, val_loader, device)
    print(f"\nTrain  MSE: {train_m['mse']:.4f}  R2: {train_m['r2']:.4f}")
    print(f"Val    MSE: {val_m['mse']:.4f}  R2: {val_m['r2']:.4f}")

    if val_m["r2"] > 0.35:
        save_artifacts(model, val_m, OUTPUT_DIR)
        print("PASS")
        sys.exit(0)
    else:
        print(f"FAIL: R2={val_m['r2']:.4f}")
        sys.exit(1)
