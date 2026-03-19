import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, accuracy_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_task_metadata():
    return {
        "task_id": "logreg_diabetes_multiclass",
        "description": "Multiclass logistic regression on Diabetes dataset (3-class binned)",
        "input_dim": 10,
        "output_dim": 3,
    }


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_dataloaders(train_ratio=0.8, batch_size=32):
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    data = load_diabetes()
    X = data.data.astype(np.float32)
    y_raw = data.target.astype(np.float32)
    p33, p66 = np.percentile(y_raw, [33.33, 66.67])
    y = np.digitize(y_raw, [p33, p66]).astype(np.int64)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, train_size=train_ratio, stratify=y, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
    )


def build_model(device=None):
    device = device or get_device()
    return nn.Sequential(nn.Linear(10, 3)).to(device)


def train(model, train_loader, val_loader, device, epochs=500, lr=0.01, weight_decay=1e-4):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(epochs):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
        if (epoch + 1) % 100 == 0:
            v = evaluate(model, val_loader, device)
            print(f"Epoch {epoch+1}/{epochs}  Acc: {v['accuracy']:.4f}  F1: {v['macro_f1']:.4f}")


def evaluate(model, data_loader, device):
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for xb, yb in data_loader:
            xb = xb.to(device)
            pred = model(xb).argmax(dim=1).cpu().numpy()
            all_pred.extend(pred)
            all_true.extend(yb.numpy())
    y_pred = np.array(all_pred)
    y_true = np.array(all_true)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def predict(model, X, device):
    model.eval()
    if not isinstance(X, torch.Tensor):
        X = torch.FloatTensor(X)
    with torch.no_grad():
        return model(X.to(device)).argmax(dim=1).cpu().numpy()


def save_artifacts(model, metrics, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))


if __name__ == "__main__":
    set_seed(42)
    device = get_device()
    print("Task: Logistic Regression (Diabetes, Multiclass)")

    train_loader, val_loader = make_dataloaders()
    model = build_model(device=device)
    train(model, train_loader, val_loader, device)

    train_m = evaluate(model, train_loader, device)
    val_m = evaluate(model, val_loader, device)
    print(f"\nTrain  Acc: {train_m['accuracy']:.4f}  Macro-F1: {train_m['macro_f1']:.4f}")
    print(f"Val    Acc: {val_m['accuracy']:.4f}  Macro-F1: {val_m['macro_f1']:.4f}")

    if val_m["accuracy"] > 0.50 and val_m["macro_f1"] > 0.50:
        save_artifacts(model, val_m, OUTPUT_DIR)
        print("PASS")
        sys.exit(0)
    else:
        print(f"FAIL: acc={val_m['accuracy']:.4f} f1={val_m['macro_f1']:.4f}")
        sys.exit(1)
