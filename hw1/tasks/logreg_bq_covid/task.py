import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
CACHE_PATH = os.path.join(SCRIPT_DIR, "covid_cache.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_task_metadata():
    return {
        "task_id": "logreg_bq_covid",
        "description": "Logistic regression classifying high vs low daily COVID case growth from BigQuery NYT data",
        "data_source": "bigquery-public-data.covid19_nyt.us_states",
        "input_dim": 4,
        "output_dim": 2,
    }


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fetch_from_bigquery(project_id=None):
    from google.cloud import bigquery

    client = bigquery.Client(project=project_id)

    sql = """
    SELECT
        state_name,
        date,
        confirmed_cases,
        deaths
    FROM `bigquery-public-data.covid19_nyt.us_states`
    WHERE confirmed_cases > 0
    ORDER BY state_name, date
    """
    print("Querying BigQuery for COVID data...")
    df = client.query(sql).to_dataframe()
    print(f"Got {len(df)} rows from BigQuery")
    return df


def prepare_features(df):
    df = df.sort_values(["state_name", "date"]).copy()
    df["new_cases"] = df.groupby("state_name")["confirmed_cases"].diff().fillna(0)
    df["new_deaths"] = df.groupby("state_name")["deaths"].diff().fillna(0)
    df["case_growth_pct"] = (
        df["new_cases"] / df["confirmed_cases"].clip(lower=1)
    ) * 100
    df["death_rate"] = (df["deaths"] / df["confirmed_cases"].clip(lower=1)) * 100

    df = df[df["new_cases"] >= 0].copy()
    df = df.dropna()

    median_growth = df["case_growth_pct"].median()
    df["high_growth"] = (df["case_growth_pct"] > median_growth).astype(int)

    df.to_csv(CACHE_PATH, index=False)
    print(f"Saved processed data to {CACHE_PATH}")
    return df


def load_data(project_id=None):
    if os.path.exists(CACHE_PATH):
        print(f"Using cached data from {CACHE_PATH}")
        return pd.read_csv(CACHE_PATH)
    raw = fetch_from_bigquery(project_id)
    return prepare_features(raw)


def make_dataloaders(train_ratio=0.8, batch_size=128, project_id=None):
    df = load_data(project_id)

    features = df[
        ["confirmed_cases", "new_cases", "death_rate", "new_deaths"]
    ].values.astype(np.float32)
    labels = df["high_growth"].values.astype(np.int64)

    X_train, X_val, y_train, y_val = train_test_split(
        features, labels, train_size=train_ratio, stratify=labels, random_state=42
    )
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_val = sc.transform(X_val)

    train_set = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_set = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True),
        DataLoader(val_set, batch_size=batch_size, shuffle=False),
    )


def build_model(device=None):
    device = device or get_device()
    return nn.Sequential(nn.Linear(4, 2)).to(device)


def train(model, train_loader, val_loader, device, epochs=300, lr=0.005):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(X), y)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 50 == 0:
            m = evaluate(model, val_loader, device)
            print(
                f"Epoch {epoch+1}/{epochs}  Acc={m['accuracy']:.4f}  F1={m['f1']:.4f}"
            )


def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            preds.extend(model(X).argmax(1).cpu().numpy())
            trues.extend(y.numpy())
    preds = np.array(preds)
    trues = np.array(trues)
    return {
        "accuracy": float(accuracy_score(trues, preds)),
        "f1": float(f1_score(trues, preds, zero_division=0)),
    }


def predict(model, X, device):
    model.eval()
    if not isinstance(X, torch.Tensor):
        X = torch.FloatTensor(X)
    with torch.no_grad():
        return model(X.to(device)).argmax(1).cpu().numpy()


def save_artifacts(model, metrics, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    with open(os.path.join(out_dir, "metrics.txt"), "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")


if __name__ == "__main__":
    set_seed(42)
    device = get_device()
    print("Task: Logistic Regression (BigQuery COVID - Case Growth)")

    train_loader, val_loader = make_dataloaders()
    model = build_model(device=device)
    train(model, train_loader, val_loader, device)

    train_m = evaluate(model, train_loader, device)
    val_m = evaluate(model, val_loader, device)
    print(f"\nTrain Acc: {train_m['accuracy']:.4f}  F1: {train_m['f1']:.4f}")
    print(f"Val   Acc: {val_m['accuracy']:.4f}  F1: {val_m['f1']:.4f}")

    if val_m["accuracy"] > 0.60 and val_m["f1"] > 0.55:
        save_artifacts(model, val_m, OUTPUT_DIR)
        print("PASS")
        sys.exit(0)
    else:
        print(f"FAIL: acc={val_m['accuracy']:.4f} f1={val_m['f1']:.4f}")
        sys.exit(1)
