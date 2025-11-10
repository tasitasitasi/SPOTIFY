
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    confusion_matrix, precision_recall_curve, roc_curve
)

from xgboost import XGBClassifier
import joblib

# -------------------- Config --------------------
DATA_PATH  = Path("data/spotty.csv")              # your cleaned dataset
MODEL_OUT  = Path("scripts/calibrated_pipeline.pkl")
THRESH_OUT = Path("scripts/model_threshold.txt")

POPULAR_THRESHOLD = 60     # label cutoff
TEST_SIZE = 0.20
RAND = 42

# Core features your app expects (must match Streamlit FEATURE_ORDER)
CORE_FEATURES = [
    "danceability","energy","musical_key","loudness","mode",
    "speechiness","acousticness","instrumentalness","liveness",
    "valence","tempo","time_signature","release_year"
]

# Optional extras (used only if present in the CSV)
OPTIONAL_FEATURES = [
    "track_duration_ms","explicit_lyrics","artist_popularity","artist_followers"
]

# Target recall for class 1 when selecting threshold (tune to taste)
TARGET_RECALL = 0.45
# ------------------------------------------------


def load_data(path: Path) -> pd.DataFrame:
    print("→ Loading data...")
    df = pd.read_csv(path)
    needed = ["track_popularity"] + CORE_FEATURES
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    # Build label
    df["is_popular"] = (df["track_popularity"] >= POPULAR_THRESHOLD).astype(int)

    # Optional type tidy
    if "explicit_lyrics" in df:
        df["explicit_lyrics"] = pd.to_numeric(df["explicit_lyrics"], errors="coerce").fillna(0).astype(int)
    if "artist_followers" in df:
        df["artist_followers"] = pd.to_numeric(df["artist_followers"], errors="coerce").fillna(0)
    if "track_duration_ms" in df:
        df["track_duration_ms"] = pd.to_numeric(df["track_duration_ms"], errors="coerce").fillna(0)

    # Ensure release_year is int
    df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce").fillna(2000).astype(int)
    # Ensure integer columns are ints
    for c in ["time_signature","mode","musical_key"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    return df


def stratified_split(X, y, test_size=0.2, random_state=42):
    print("→ Stratified split (80/20)...")
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    (train_idx, test_idx) = next(sss.split(X, y))
    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]


def choose_threshold_with_target_recall(y_true, y_prob, target_recall=0.45):
    """Pick threshold meeting target recall with best accuracy; fallback to best F1."""
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    # precision_recall_curve returns len(thr) = len(rec) - 1
    thr = np.append(thr, 1.0)

    mask = rec >= target_recall
    if mask.any():
        best_acc = -1.0
        best_thr = 0.5
        for t in thr[mask]:
            y_pred = (y_prob >= t).astype(int)
            acc = accuracy_score(y_true, y_pred)
            if acc > best_acc:
                best_acc = acc
                best_thr = float(t)
        return best_thr, "target_recall"
    else:
        # fallback: best F1
        f1 = (2 * prec * rec) / np.clip(prec + rec, 1e-12, None)
        idx = int(np.nanargmax(f1))
        chosen = float(thr[idx])
        return chosen, "best_f1"


def main():
    df = load_data(DATA_PATH)

    # Decide final FEATURES = core + any optional present
    features = CORE_FEATURES + [c for c in OPTIONAL_FEATURES if c in df.columns]
    print("Using features:", features)

    X = df[features]
    y = df["is_popular"]

    X_train, X_test, y_train, y_test = stratified_split(X, y, TEST_SIZE, RAND)

    # Class imbalance ratio for XGBoost
    neg, pos = y_train.value_counts()
    ratio = neg / max(pos, 1)
    print(f"Class balance (train): neg={neg}, pos={pos}, scale_pos_weight={ratio:.2f}")

    print("→ Training XGBoost (with early stopping)...")
    xgb = XGBClassifier(
        n_estimators=1200,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=5,
        gamma=0.0,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.5,
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=ratio * 0.7,   # tempered imbalance to recover accuracy
        random_state=RAND,
        n_jobs=-1,
        tree_method="hist"
    )

    xgb.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False
    )

    # Predictions
    y_prob = xgb.predict_proba(X_test)[:, 1]
    y_pred_05 = (y_prob >= 0.50).astype(int)

    acc = accuracy_score(y_test, y_pred_05)
    auc = roc_auc_score(y_test, y_prob)

    print(f"\nAccuracy (thr=0.50): {acc:.3f}")
    print(f"ROC-AUC: {auc:.3f}")
    print("\nClassification Report (thr=0.50):")
    print(classification_report(y_test, y_pred_05, digits=3))
    print("Confusion Matrix (thr=0.50):\n", confusion_matrix(y_test, y_pred_05))

    # Choose threshold (target recall, fallback best F1)
    best_thr, how = choose_threshold_with_target_recall(y_test, y_prob, TARGET_RECALL)
    y_pred_best = (y_prob >= best_thr).astype(int)
    acc_best = accuracy_score(y_test, y_pred_best)

    print(f"\nChosen threshold = {best_thr:.3f}  (via {how})")
    print(f"Accuracy @ chosen thr: {acc_best:.3f}")
    print("Classification Report @ chosen thr:")
    print(classification_report(y_test, y_pred_best, digits=3))
    print("Confusion Matrix @ chosen thr:\n", confusion_matrix(y_test, y_pred_best))

    # Save model + threshold
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(xgb, MODEL_OUT)
    THRESH_OUT.parent.mkdir(parents=True, exist_ok=True)
    THRESH_OUT.write_text(f"{best_thr:.4f}")

    print(f"\nSaved model → {MODEL_OUT}")
    print(f"Saved threshold → {THRESH_OUT}  (use as Streamlit default)")

    # Print feature importances
    importances = pd.Series(xgb.feature_importances_, index=features).sort_values(ascending=False)
    print("\nTop feature importances:\n", importances.head(15))


if __name__ == "__main__":
    main()
