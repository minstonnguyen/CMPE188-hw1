import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
CACHE_CSV = os.path.join(SCRIPT_DIR, "natality_cache.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_task_metadata():
    return {
        "task_id": "linreg_bq_natality",
        "description": "Linear regression predicting baby birth weight from BigQuery natality data",
        "data_source": "bigquery-public-data.samples.natality",
        "input_dim": 5,
        "output_dim": 1,
    }


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_from_bigquery(project_id=None):
    from google.cloud import bigquery

    client = bigquery.Client(project=project_id)

    sql = """
    SELECT
        weight_pounds,
        mother_age,
        gestation_weeks,
        plurality,
        is_male,
        cigarette_use
    FROM `bigquery-public-data.samples.natality`
    WHERE weight_pounds IS NOT NULL
        AND mother_age IS NOT NULL
        AND gestation_weeks IS NOT NULL
        AND NOT IS_NAN(weight_pounds)
        AND NOT IS_NAN(gestation_weeks)
        AND gestation_weeks BETWEEN 20 AND 45
        AND weight_pounds > 0
    LIMIT 5000
    """
    print("Running BigQuery query...")
    df = client.query(sql).to_dataframe()
    print(f"Loaded {len(df)} rows from BigQuery")

    df["is_male"] = df["is_male"].astype(int)
    df["cigarette_use"] = df["cigarette_use"].fillna(0).astype(int)

    df.to_csv(CACHE_CSV, index=False)
    print(f"Cached data to {CACHE_CSV}")
    return df


def load_data(project_id=None):
    if os.path.exists(CACHE_CSV):
        print(f"Loading cached data from {CACHE_CSV}")
        return pd.read_csv(CACHE_CSV)
    return load_from_bigquery(project_id)


def make_dataloaders(train_ratio=0.8, batch_size=64, project_id=None):
    df = load_data(project_id)

    feature_cols = [
        "mother_age",
        "gestation_weeks",
        "plurality",
        "is_male",
        "cigarette_use",
    ]
    X = df[feature_cols].values.astype(np.float32)
    y = df["weight_pounds"].values.astype(np.float32).reshape(-1, 1)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, train_size=train_ratio, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def build_model(device=None):
    device = device or get_device()
    return nn.Sequential(nn.Linear(5, 1)).to(device)


def train(model, train_loader, val_loader, device, epochs=300, lr=0.05):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        total_loss = 0.0
        count = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            count += 1
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/count:.4f}")


def evaluate(model, data_loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in data_loader:
            xb = xb.to(device)
            preds.append(model(xb).cpu())
            targets.append(yb)
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
    X = X.to(device)
    with torch.no_grad():
        return model(X).cpu().numpy()


def save_artifacts(model, metrics, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
    with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
        f.write(f"MSE: {metrics['mse']}\nR2: {metrics['r2']}\n")


if __name__ == "__main__":
    set_seed(42)
    device = get_device()
    print("Task: Linear Regression (BigQuery Natality - Birth Weight)")
    print(f"Device: {device}")

    train_loader, val_loader = make_dataloaders()
    model = build_model(device=device)
    train(model, train_loader, val_loader, device, epochs=300, lr=0.05)

    train_m = evaluate(model, train_loader, device)
    val_m = evaluate(model, val_loader, device)

    print(f"\nTrain MSE: {train_m['mse']:.4f}, R2: {train_m['r2']:.4f}")
    print(f"Val   MSE: {val_m['mse']:.4f}, R2: {val_m['r2']:.4f}")

    if val_m["r2"] > 0.15 and val_m["mse"] < 2.0:
        save_artifacts(model, val_m, OUTPUT_DIR)
        print("\nPASS")
        sys.exit(0)
    else:
        print(f"\nFAIL: R2={val_m['r2']:.4f}, MSE={val_m['mse']:.4f}")
        sys.exit(1)
