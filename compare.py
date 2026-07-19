# compare.py  — Step 5: Compare Centralized vs Federated Learning

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

CENT_LOG = "logs/centralized_metrics.json"
FL_LOG   = "logs/fl_metrics.json"
OUT_DIR  = "plots"

os.makedirs(OUT_DIR, exist_ok=True)


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


cent = load_json(CENT_LOG)
fl   = load_json(FL_LOG)

if not cent:
    print(f"❌ Centralized log not found: {CENT_LOG}")
    print("   Run:  python centralized_train.py")
if not fl:
    print(f"❌ FL log not found: {FL_LOG}")
    print("   Run:  python -m fl.simulation")

if not cent or not fl:
    raise SystemExit(1)

# ── Extract metrics ────────────────────────────────────────────────────────────
cent_epochs   = [e["epoch"]    for e in cent["epochs"]]
cent_auc      = [e["auc"]      for e in cent["epochs"]]
cent_acc      = [e["accuracy"] for e in cent["epochs"]]
cent_f1       = [e["f1"]       for e in cent["epochs"]]

fl_rounds     = [r["round"]        for r in fl["rounds"]]
fl_auc        = [r["avg_auc"]      for r in fl["rounds"]]
fl_acc        = [r["avg_accuracy"] for r in fl["rounds"]]
fl_f1         = [r["avg_f1"]       for r in fl["rounds"]]

# Per-client FL metrics
num_clients = len(fl["rounds"][0]["client_metrics"]) if fl["rounds"] else 0
client_aucs = {i: [] for i in range(num_clients)}
for r in fl["rounds"]:
    for i, cm in enumerate(r["client_metrics"]):
        client_aucs[i].append(cm["auc"])

split_mode = fl.get("split_mode", "non_iid").upper()

# ── Plot ───────────────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-darkgrid")
fig = plt.figure(figsize=(18, 12))
fig.suptitle("Centralized vs Federated Learning — Skin Lesion Classification",
             fontsize=15, fontweight="bold", y=0.98)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

# ── AUC comparison ────────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(cent_epochs, cent_auc, "b-o", label="Centralized", linewidth=2, markersize=5)
ax1.plot(fl_rounds,   fl_auc,   "r-s", label=f"FL ({split_mode})", linewidth=2, markersize=5)
ax1.set_title("AUC vs Epoch/Round")
ax1.set_xlabel("Epoch / Round")
ax1.set_ylabel("AUC")
ax1.legend()
ax1.set_ylim(0, 1)

# ── Accuracy comparison ───────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(cent_epochs, cent_acc, "b-o", label="Centralized", linewidth=2, markersize=5)
ax2.plot(fl_rounds,   fl_acc,   "r-s", label=f"FL ({split_mode})", linewidth=2, markersize=5)
ax2.set_title("Accuracy vs Epoch/Round")
ax2.set_xlabel("Epoch / Round")
ax2.set_ylabel("Accuracy")
ax2.legend()
ax2.set_ylim(0, 1)

# ── F1 comparison ─────────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(cent_epochs, cent_f1, "b-o", label="Centralized", linewidth=2, markersize=5)
ax3.plot(fl_rounds,   fl_f1,  "r-s", label=f"FL ({split_mode})", linewidth=2, markersize=5)
ax3.set_title("F1 Score vs Epoch/Round")
ax3.set_xlabel("Epoch / Round")
ax3.set_ylabel("F1 (Macro)")
ax3.legend()
ax3.set_ylim(0, 1)

# ── Per-client AUC (FL) ───────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
colors = plt.cm.Set1.colors
for i in range(num_clients):
    ax4.plot(fl_rounds, client_aucs[i], "-o",
             color=colors[i % len(colors)],
             label=f"Client {i}", linewidth=2, markersize=5)
ax4.set_title(f"FL — Per-Client AUC ({split_mode})")
ax4.set_xlabel("Round")
ax4.set_ylabel("AUC")
ax4.legend()
ax4.set_ylim(0, 1)

# ── Final metric bar chart ────────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
metrics = ["AUC", "Accuracy", "F1"]
cent_final = [cent_auc[-1], cent_acc[-1], cent_f1[-1]] if cent_auc else [0, 0, 0]
fl_final   = [fl_auc[-1],   fl_acc[-1],  fl_f1[-1]]   if fl_auc   else [0, 0, 0]

x      = np.arange(len(metrics))
width  = 0.35
bars1  = ax5.bar(x - width/2, cent_final, width, label="Centralized", color="steelblue")
bars2  = ax5.bar(x + width/2, fl_final,   width, label=f"FL ({split_mode})", color="salmon")
ax5.set_title("Final Round Comparison")
ax5.set_xticks(x)
ax5.set_xticklabels(metrics)
ax5.set_ylabel("Score")
ax5.set_ylim(0, 1.1)
ax5.legend()
for bar in bars1 + bars2:
    h = bar.get_height()
    ax5.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width()/2, h),
                 xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

# ── Summary table (text) ──────────────────────────────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis("off")
summary = [
    ["Metric",     "Centralized",         f"FL ({split_mode})"],
    ["Best AUC",   f"{max(cent_auc):.4f}", f"{max(fl_auc):.4f}"],
    ["Best Acc",   f"{max(cent_acc):.4f}", f"{max(fl_acc):.4f}"],
    ["Best F1",    f"{max(cent_f1):.4f}",  f"{max(fl_f1):.4f}"],
    ["Epochs/Rds", str(len(cent_epochs)),  str(len(fl_rounds))],
]
table = ax6.table(cellText=summary, cellLoc="center", loc="center",
                  colWidths=[0.3, 0.35, 0.35])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)
ax6.set_title("Summary", fontsize=12, fontweight="bold")

out_path = os.path.join(OUT_DIR, "comparison.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\n✅ Comparison plot saved to {out_path}")
plt.show()  # Display chart directly on screen

# ── Console summary ───────────────────────────────────────────────────────────
print("\n" + "═" * 50)
print("  FINAL COMPARISON SUMMARY")
print("═" * 50)
for metric, c, f in zip(metrics, cent_final, fl_final):
    diff = f - c
    arrow = "▲" if diff >= 0 else "▼"
    print(f"  {metric:8s}: Centralized={c:.4f}  FL={f:.4f}  {arrow}{abs(diff):.4f}")
print("═" * 50)
