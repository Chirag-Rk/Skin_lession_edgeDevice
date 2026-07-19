import os
import torch

# ======================
# CONFIG (CPU OPTIMIZED)
# ======================
CFG = {
    "image_size":     128,      # reduced for CPU
    "batch_size":     8,        # smaller batch
    "epochs":         1,        # FL uses per-round training
    "num_classes":    7,
    "lr":             1e-4,
    "use_amp":        False,    # disabled on CPU
    "checkpoint_dir": "./checkpoints",
    "grad_clip":      1.0,
    "pretrained":     True,     # set False to skip HuggingFace download
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # CUDA on Jetson, fallback to CPU

os.makedirs(CFG["checkpoint_dir"], exist_ok=True)

CKPT_PATH = os.path.join(CFG["checkpoint_dir"], "best_model.pt")

# ======================
# LABEL MAP
# ======================
label_map = {
    "nv":    0,
    "mel":   1,
    "bkl":   2,
    "bcc":   3,
    "akiec": 4,
    "df":    5,
    "vasc":  6,
}

# ======================
# DATA LOADING (LAZY)
# Dataset is loaded lazily — only when explicitly called.
# This prevents import-time crashes when CSVs don't exist yet.
# ======================
DATA_DIR = "data/HAM10000"

def load_dataframes():
    """
    Load and prepare the HAM10000 dataframe.
    Call this explicitly in scripts — NOT at import time.
    Returns: (full_df, train_df, val_df)
    """
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split

    csv_path = os.path.join(DATA_DIR, "HAM10000_metadata.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"\n❌ Dataset not found at: {csv_path}\n"
            "   Download HAM10000 and place it in data/HAM10000/\n"
            "   See README.md for instructions."
        )

    df = pd.read_csv(csv_path)
    df["label"] = df["dx"].map(label_map)

    img1 = os.path.join(DATA_DIR, "HAM10000_images_part_1")
    img2 = os.path.join(DATA_DIR, "HAM10000_images_part_2")

    def get_path(x):
        p1 = os.path.join(img1, x + ".jpg")
        p2 = os.path.join(img2, x + ".jpg")
        return p1 if os.path.exists(p1) else p2

    df["path"] = df["image_id"].apply(get_path)

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["label"],
        random_state=42,
    )

    print(f"  Total samples : {len(df)}")
    print(f"  Train samples : {len(train_df)}")
    print(f"  Val samples   : {len(val_df)}")

    return df, train_df, val_df


# Legacy module-level aliases (populated only if dataset present, silently skipped otherwise)
try:
    import pandas as pd
    from sklearn.model_selection import train_test_split

    _csv = os.path.join(DATA_DIR, "HAM10000_metadata.csv")
    if os.path.exists(_csv):
        import numpy as np
        df = pd.read_csv(_csv)
        df["label"] = df["dx"].map(label_map)
        img1 = os.path.join(DATA_DIR, "HAM10000_images_part_1")
        img2 = os.path.join(DATA_DIR, "HAM10000_images_part_2")
        df["path"] = df["image_id"].apply(
            lambda x: os.path.join(img1, x + ".jpg")
            if os.path.exists(os.path.join(img1, x + ".jpg"))
            else os.path.join(img2, x + ".jpg")
        )
        train_df, val_df = train_test_split(
            df, test_size=0.2, stratify=df["label"], random_state=42
        )
    else:
        df = train_df = val_df = None
except Exception:
    df = train_df = val_df = None