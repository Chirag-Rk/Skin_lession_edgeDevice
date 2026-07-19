# fl/simulation.py  — Step 1 + 2 + 3: Run FL with 3 clients, 3 rounds, full logging

import flwr as fl
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.dataset import SkinDataset, get_tf
from src.model import MobileNetAttentionModel
from src.config import CFG, DEVICE


# =============================================================================
# DATA LOADING
# =============================================================================
DATA_DIR = "data/HAM10000"

df = pd.read_csv(os.path.join(DATA_DIR, "HAM10000_metadata.csv"))

label_map = {"nv": 0, "mel": 1, "bkl": 2, "bcc": 3, "akiec": 4, "df": 5, "vasc": 6}
df["label"] = df["dx"].map(label_map)

img1 = os.path.join(DATA_DIR, "HAM10000_images_part_1")
img2 = os.path.join(DATA_DIR, "HAM10000_images_part_2")


def get_path(x):
    p1 = os.path.join(img1, x + ".jpg")
    p2 = os.path.join(img2, x + ".jpg")
    return p1 if os.path.exists(p1) else p2


df["path"] = df["image_id"].apply(get_path)


# =============================================================================
# CLIENT DATA SPLITS
# =============================================================================

def split_iid(df, num_clients=3):
    """IID split: random equal partition across clients."""
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    chunks = np.array_split(df, num_clients)
    return chunks


def split_non_iid(df, client_id, num_clients=3):
    """Non-IID split: each client gets distinct label groups."""
    label_groups = {
        0: [0, 1],       # nv, mel
        1: [2, 3],       # bkl, bcc
        2: [4, 5, 6],    # akiec, df, vasc
    }
    labels = label_groups[client_id % num_clients]
    return df[df["label"].isin(labels)]


def split_extreme_non_iid(df, client_id, num_clients=3):
    """Extreme Non-IID: each client gets only 1–2 dominant classes."""
    labels_per_client = {0: [0], 1: [1], 2: [2]}  # single dominant class
    labels = labels_per_client.get(client_id, [0])
    majority = df[df["label"].isin(labels)]
    minority = df[~df["label"].isin(labels)].sample(
        frac=0.05, random_state=42
    )   # tiny noise
    return pd.concat([majority, minority]).sample(frac=1, random_state=42)


SPLIT_MODE = os.environ.get("SPLIT_MODE", "non_iid")   # iid | non_iid | extreme


def get_client_data(df, client_id, num_clients=3):
    # STEP 10: Domain Shift Experiment (ISIC Dataset)
    if SPLIT_MODE == "domain_shift" and client_id == 2:
        # Load independent ISIC dataset for client 2
        isic_path = os.path.join("data", "ISIC", "ISIC_metadata.csv")
        if os.path.exists(isic_path):
            print(f"\n[Domain Shift] Client {client_id} loading ISIC dataset ...")
            isic_df = pd.read_csv(isic_path)
            # Adapt label mapping as needed
            isic_df["label"] = 1 # Example formatting 
            df_client = isic_df
        else:
            print(f"\n⚠️ ISIC dataset not found at {isic_path}. Falling back to HAM10000 non-iid.")
            df_client = split_non_iid(df, client_id, num_clients)
    
    elif SPLIT_MODE == "iid":
        chunks = split_iid(df, num_clients)
        df_client = chunks[client_id]
    elif SPLIT_MODE == "extreme":
        df_client = split_extreme_non_iid(df, client_id, num_clients)
    else:
        df_client = split_non_iid(df, client_id, num_clients)

    # Ensure at least 2 classes for stratified split
    counts = df_client["label"].value_counts()
    valid_labels = counts[counts >= 2].index
    df_client = df_client[df_client["label"].isin(valid_labels)]

    try:
        train_df, val_df = train_test_split(
            df_client,
            test_size=0.2,
            stratify=df_client["label"],
            random_state=42,
        )
    except ValueError:
        train_df, val_df = train_test_split(df_client, test_size=0.2, random_state=42)

    print(f"\n{'─'*40}")
    print(f"  Client {client_id}  |  Split: {SPLIT_MODE.upper()}")
    print(f"  Train: {len(train_df)}   Val: {len(val_df)}")
    print(f"  Label distribution:\n{df_client['label'].value_counts().to_string()}")
    print(f"{'─'*40}")

    train_loader = DataLoader(
        SkinDataset(train_df, get_tf("train")),
        batch_size=CFG["batch_size"],
        shuffle=True,
    )
    val_loader = DataLoader(
        SkinDataset(val_df, get_tf("val")),
        batch_size=CFG["batch_size"],
        shuffle=False,
    )
    return train_loader, val_loader


# =============================================================================
# ROUND-LEVEL LOGGING STRATEGY
# =============================================================================

class LoggingStrategy(fl.server.strategy.FedAvg):
    """FedAvg + per-round metric logging."""

    def __init__(self, num_rounds, log_path="logs/fl_metrics.json", **kwargs):
        super().__init__(**kwargs)
        self.round_logs = []
        self.num_rounds = num_rounds
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def aggregate_evaluate(self, server_round, results, failures):
        aggregated = super().aggregate_evaluate(server_round, results, failures)

        if not results:
            return aggregated

        # Collect per-client metrics
        client_metrics = []
        for _, eval_res in results:
            m = eval_res.metrics
            client_metrics.append({
                "accuracy": round(m.get("accuracy", 0.0), 4),
                "auc":      round(m.get("auc", 0.0), 4),
                "f1":       round(m.get("f1", 0.0), 4),
            })

        # Aggregate means
        avg_acc = float(np.mean([m["accuracy"] for m in client_metrics]))
        avg_auc = float(np.mean([m["auc"]      for m in client_metrics]))
        avg_f1  = float(np.mean([m["f1"]       for m in client_metrics]))

        round_entry = {
            "round":          server_round,
            "avg_accuracy":   round(avg_acc, 4),
            "avg_auc":        round(avg_auc, 4),
            "avg_f1":         round(avg_f1, 4),
            "client_metrics": client_metrics,
        }
        self.round_logs.append(round_entry)

        print(f"\n{'='*50}")
        print(f"  ROUND {server_round}/{self.num_rounds} — AGGREGATED METRICS")
        print(f"  Avg Accuracy : {avg_acc:.4f}")
        print(f"  Avg AUC      : {avg_auc:.4f}")
        print(f"  Avg F1       : {avg_f1:.4f}")
        for i, cm in enumerate(client_metrics):
            print(f"  Client {i}     : Acc={cm['accuracy']}  AUC={cm['auc']}  F1={cm['f1']}")
        print(f"{'='*50}\n")

        # Save logs after every round
        with open(self.log_path, "w") as f:
            json.dump({
                "split_mode": SPLIT_MODE,
                "timestamp":  datetime.now().isoformat(),
                "rounds":     self.round_logs,
            }, f, indent=2)

        return aggregated


# =============================================================================
# CLIENT FUNCTION
# =============================================================================
from fl.client import SkinClient   # keep here to avoid circular import


def client_fn(cid: str):
    cid_int = int(cid)
    model = MobileNetAttentionModel().to(DEVICE)
    train_loader, val_loader = get_client_data(df, cid_int)
    return SkinClient(model, train_loader, val_loader, client_id=cid_int)


# =============================================================================
# MAIN SIMULATION
# =============================================================================

NUM_CLIENTS = int(os.environ.get("CLIENTS", 3))
NUM_ROUNDS  = int(os.environ.get("ROUNDS", 3))

if __name__ == "__main__":
    strategy = LoggingStrategy(
        num_rounds=NUM_ROUNDS,
        log_path="logs/fl_metrics.json",
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
    )

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )

    print("\n[OK] FL Simulation complete. Logs saved to logs/fl_metrics.json")