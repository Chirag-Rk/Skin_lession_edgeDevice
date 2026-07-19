# plot_fl.py  — Step 8: Visualize FL round-wise metrics and client performance

import os
import json
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

FL_LOG  = "logs/fl_metrics.json"
OUT_DIR = "plots"
os.makedirs(OUT_DIR, exist_ok=True)

if not os.path.exists(FL_LOG):
    raise FileNotFoundError(f"❌ FL log not found: {FL_LOG}\n   Run:  python -m fl.simulation")

with open(FL_LOG) as f:
    data = json.load(f)

rounds      = [r["round"]        for r in data["rounds"]]
avg_auc     = [r["avg_auc"]      for r in data["rounds"]]
avg_acc     = [r["avg_accuracy"] for r in data["rounds"]]
avg_f1      = [r["avg_f1"]       for r in data["rounds"]]
split_mode  = data.get("split_mode", "non_iid").upper()

num_clients = len(data["rounds"][0]["client_metrics"]) if data["rounds"] else 0
client_data = {
    i: {
        "auc":      [r["client_metrics"][i]["auc"]      for r in data["rounds"]],
        "accuracy": [r["client_metrics"][i]["accuracy"] for r in data["rounds"]],
        "f1":       [r["client_metrics"][i]["f1"]       for r in data["rounds"]],
    }
    for i in range(num_clients)
}

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
})
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]

fig = plt.figure(figsize=(18, 14))
fig.suptitle(f"Federated Learning Metrics — {split_mode} Split",
             fontsize=16, fontweight="bold", y=0.99)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# ── Global AUC ────────────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
ax.plot(rounds, avg_auc, "o-", color="#2196F3", linewidth=2.5, markersize=7, label="Avg AUC")
ax.fill_between(rounds, avg_auc, alpha=0.15, color="#2196F3")
ax.set_title("Global AUC vs Round")
ax.set_xlabel("Round"); ax.set_ylabel("AUC")
ax.set_ylim(0, 1); ax.legend()

# ── Global Accuracy ───────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 1])
ax.plot(rounds, avg_acc, "s-", color="#4CAF50", linewidth=2.5, markersize=7, label="Avg Accuracy")
ax.fill_between(rounds, avg_acc, alpha=0.15, color="#4CAF50")
ax.set_title("Global Accuracy vs Round")
ax.set_xlabel("Round"); ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1); ax.legend()

# ── Global F1 ─────────────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 2])
ax.plot(rounds, avg_f1, "^-", color="#FF9800", linewidth=2.5, markersize=7, label="Avg F1")
ax.fill_between(rounds, avg_f1, alpha=0.15, color="#FF9800")
ax.set_title("Global F1 vs Round")
ax.set_xlabel("Round"); ax.set_ylabel("F1 (Macro)")
ax.set_ylim(0, 1); ax.legend()

# ── Per-client AUC ────────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 0])
for i in range(num_clients):
    ax.plot(rounds, client_data[i]["auc"], "o-",
            color=COLORS[i % len(COLORS)], linewidth=2, markersize=6, label=f"Client {i}")
ax.set_title("Per-Client AUC vs Round")
ax.set_xlabel("Round"); ax.set_ylabel("AUC")
ax.set_ylim(0, 1); ax.legend()

# ── Per-client Accuracy ───────────────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 1])
for i in range(num_clients):
    ax.plot(rounds, client_data[i]["accuracy"], "s-",
            color=COLORS[i % len(COLORS)], linewidth=2, markersize=6, label=f"Client {i}")
ax.set_title("Per-Client Accuracy vs Round")
ax.set_xlabel("Round"); ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1); ax.legend()

# ── Per-client F1 ─────────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 2])
for i in range(num_clients):
    ax.plot(rounds, client_data[i]["f1"], "^-",
            color=COLORS[i % len(COLORS)], linewidth=2, markersize=6, label=f"Client {i}")
ax.set_title("Per-Client F1 vs Round")
ax.set_xlabel("Round"); ax.set_ylabel("F1 (Macro)")
ax.set_ylim(0, 1); ax.legend()

# ── AUC Variance across clients ───────────────────────────────────────────────
ax = fig.add_subplot(gs[2, 0])
auc_matrix = np.array([client_data[i]["auc"] for i in range(num_clients)])  # (n_clients, n_rounds)
auc_mean   = auc_matrix.mean(axis=0)
auc_std    = auc_matrix.std(axis=0)
ax.plot(rounds, auc_mean, "o-", color="#9C27B0", linewidth=2.5, label="Mean AUC")
ax.fill_between(rounds, auc_mean - auc_std, auc_mean + auc_std, alpha=0.2, color="#9C27B0", label="±1 Std")
ax.set_title("AUC Mean ± Std (Client Variance)")
ax.set_xlabel("Round"); ax.set_ylabel("AUC")
ax.set_ylim(0, 1); ax.legend()

# ── Final round bar chart per client ─────────────────────────────────────────
ax = fig.add_subplot(gs[2, 1])
last_round_metrics = data["rounds"][-1]["client_metrics"]
x = np.arange(num_clients)
w = 0.25
bars_auc = ax.bar(x - w, [m["auc"]      for m in last_round_metrics], w, label="AUC",      color="#2196F3")
bars_acc = ax.bar(x,     [m["accuracy"] for m in last_round_metrics], w, label="Accuracy", color="#4CAF50")
bars_f1  = ax.bar(x + w, [m["f1"]       for m in last_round_metrics], w, label="F1",       color="#FF9800")
ax.set_title(f"Final Round Metrics per Client")
ax.set_xticks(x); ax.set_xticklabels([f"Client {i}" for i in range(num_clients)])
ax.set_ylabel("Score"); ax.set_ylim(0, 1.15); ax.legend()
for bar in list(bars_auc) + list(bars_acc) + list(bars_f1):
    h = bar.get_height()
    ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width()/2, h),
                xytext=(0, 2), textcoords="offset points", ha="center", fontsize=7)

# ── Summary table ─────────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[2, 2])
ax.axis("off")
final_metrics = data["rounds"][-1]
rows = [
    ["Metric",        "Value"],
    ["Split Mode",    split_mode],
    ["Num Rounds",    str(len(rounds))],
    ["Num Clients",   str(num_clients)],
    ["Final AUC",     f"{final_metrics['avg_auc']:.4f}"],
    ["Final Acc",     f"{final_metrics['avg_accuracy']:.4f}"],
    ["Final F1",      f"{final_metrics['avg_f1']:.4f}"],
    ["Best AUC",      f"{max(avg_auc):.4f}"],
]
table = ax.table(cellText=rows, cellLoc="center", loc="center",
                 colWidths=[0.4, 0.5])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.8)
ax.set_title("FL Summary", fontsize=11, fontweight="bold")

out_path = os.path.join(OUT_DIR, "fl_metrics.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"✅ FL metrics plot saved to {out_path}")
plt.show()  # Display chart directly on screen
