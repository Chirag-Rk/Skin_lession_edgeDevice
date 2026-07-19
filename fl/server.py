import flwr as fl
import numpy as np
import json
import os
from datetime import datetime

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
                "split_mode": os.environ.get("SPLIT_MODE", "unknown"),
                "timestamp":  datetime.now().isoformat(),
                "rounds":     self.round_logs,
            }, f, indent=2)

        return aggregated


if __name__ == "__main__":
    NUM_ROUNDS = int(os.environ.get("ROUNDS", 1))
    MIN_CLIENTS = int(os.environ.get("CLIENTS", 2))

    strategy = LoggingStrategy(
        num_rounds=NUM_ROUNDS,
        min_fit_clients=MIN_CLIENTS,
        min_available_clients=MIN_CLIENTS,
        min_evaluate_clients=MIN_CLIENTS,
    )

    print(f"[OK] Starting Federated Learning Server (Rounds: {NUM_ROUNDS}, Min Clients: {MIN_CLIENTS})")
    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )