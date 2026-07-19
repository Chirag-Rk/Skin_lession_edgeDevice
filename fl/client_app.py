# fl/client_app.py  — Multi-terminal setup (Option 2)
# Usage: python fl/client_app.py <client_id>
#   e.g. python fl/client_app.py 0

import sys
import os

# Ensure the root project directory is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import flwr as fl
import pandas as pd

from src.model import MobileNetAttentionModel
from src.config import CFG, DEVICE
from fl.client import SkinClient
from fl.simulation import get_client_data, df   # reuse data loading & split logic


def start_client(cid: int):
    model = MobileNetAttentionModel(num_classes=CFG["num_classes"]).to(DEVICE)
    train_loader, val_loader = get_client_data(df, cid)
    client = SkinClient(model, train_loader, val_loader, client_id=cid)

    print(f"\n🔌 Client {cid} connecting to server at localhost:8080 ...")
    fl.client.start_numpy_client(
        server_address="localhost:8080",
        client=client,
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fl/client_app.py <client_id>")
        sys.exit(1)
    cid = int(sys.argv[1])
    start_client(cid)