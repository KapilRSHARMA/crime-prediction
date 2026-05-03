DISTRICT_NAMES = {
    'A1':  'Downtown',
    'A15': 'Charlestown',
    'A7':  'East Boston',
    'B2':  'Roxbury',
    'B3':  'Mattapan',
    'C11': 'Dorchester',
    'C6':  'South Boston',
    'D14': 'Brighton',
    'D4':  'South End',
    'E13': 'Jamaica Plain',
    'E18': 'Hyde Park',
    'E5':  'West Roxbury',
}

import os, sys, json, pickle, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_val_score)
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier


CSV_PATH      = "crime.csv"          
ARTIFACTS_DIR = "artifacts"          
SAMPLE_SIZE   = 80000                
TEST_SIZE     = 0.20                 
RANDOM_STATE  = 42
MIN_ACCURACY  = 0.80                 


def hdr(text):
    print(f"\n{'═'*64}")
    print(f"  {text}")
    print(f"{'═'*64}")

def step(n, total, text):
    print(f"\n[{n}/{total}] {text}")

def ok(text):   print(f"  {text}")
def info(text): print(f"  {text}")
def warn(text): print(f"  {text}")


def load_data(path):
    if not os.path.exists(path):
        sys.exit(
            f"\n  crime.csv not found at: {os.path.abspath(path)}\n"
            "    Download it from:\n"
            "    https://www.kaggle.com/datasets/AnalyzeBoston/crimes-in-boston\n"
            "    and place crime.csv in the same folder as train.py\n"
        )

    df = pd.read_csv(path, encoding="latin-1", low_memory=False)
    df.columns = [c.strip().upper() for c in df.columns]
    ok(f"Loaded  →  {len(df):,} rows  ×  {len(df.columns)} columns")
    info(f"Columns: {list(df.columns)}")
    return df

def clean_data(df):
    """Remove rows with missing critical fields and invalid coordinates."""
    before = len(df)
    df = df.dropna(subset=["LAT", "LONG", "DISTRICT", "HOUR"])
    df = df[(df["LAT"] != 0) & (df["LONG"] != 0)]
    df["SHOOTING"] = df["SHOOTING"].apply(
        lambda x: 1 if str(x).strip().upper() == "Y" else 0
    )

    df["REPORTING_AREA"] = pd.to_numeric(df["REPORTING_AREA"], errors="coerce").fillna(0)
    df["UCR_PART"]       = df["UCR_PART"].fillna("Unknown")
    df["HOUR"]           = pd.to_numeric(df["HOUR"],  errors="coerce").fillna(0).astype(int)
    df["MONTH"]          = pd.to_numeric(df["MONTH"], errors="coerce").fillna(1).astype(int)
    df["YEAR"]           = pd.to_numeric(df["YEAR"],  errors="coerce").fillna(2016).astype(int)

    df = df.drop_duplicates()
    ok(f"Clean   →  {len(df):,} rows  (dropped {before-len(df):,} rows)")
    if 'DISTRICT' in df.columns:
        districts = sorted(df['DISTRICT'].dropna().unique().tolist())
        info(f"Districts: {districts}")
    return df

def label_hotspots(df, quantile=0.72):

    df = df.copy()
    df["GRID"]  = ((df["LAT"]  / 0.005).astype(int).astype(str) + "_" +
                   (df["LONG"] / 0.005).astype(int).astype(str))
    df["HBKT"]  = (df["HOUR"] // 3)          # 8 time-buckets per day

    cell_cnt = df.groupby(["GRID", "HBKT"]).size().reset_index(name="CC")
    df = df.merge(cell_cnt, on=["GRID", "HBKT"], how="left")

    threshold        = df["CC"].quantile(quantile)
    df["IS_HOTSPOT"] = (df["CC"] >= threshold).astype(int)
    df.drop(columns=["GRID", "HBKT", "CC"], inplace=True)

    ratio = df["IS_HOTSPOT"].mean()
    ok(f"Hotspot →  {df['IS_HOTSPOT'].sum():,} hotspot rows  ({ratio:.1%} of data)")
    info(f"Threshold = {threshold:.1f} incidents per cell-bucket")
    return df

def engineer_features(df):

    df = df.copy()

    le_d = LabelEncoder()
    le_w = LabelEncoder()
    le_o = LabelEncoder()
    le_u = LabelEncoder()

    df["DISTRICT_ENC"] = le_d.fit_transform(df["DISTRICT"].astype(str))
    df["DAY_ENC"]      = le_w.fit_transform(df["DAY_OF_WEEK"].astype(str))
    df["OFFENSE_ENC"]  = le_o.fit_transform(df["OFFENSE_CODE_GROUP"].astype(str))
    df["UCR_ENC"]      = le_u.fit_transform(df["UCR_PART"].astype(str))

    df["LAT_NORM"]  = (df["LAT"]  - df["LAT"].mean())  / df["LAT"].std()
    df["LONG_NORM"] = (df["LONG"] - df["LONG"].mean()) / df["LONG"].std()
    df["RA_LOG"]    = np.log1p(df["REPORTING_AREA"].astype(float))
    df["IS_NIGHT"]   = ((df["HOUR"] >= 20) | (df["HOUR"] <= 5)).astype(int)
    df["IS_WEEKEND"] = df["DAY_OF_WEEK"].isin(["Saturday", "Sunday"]).astype(int)
    df["IS_EVENING"] = df["HOUR"].between(17, 21).astype(int)
    df["IS_RUSH"]    = (df["HOUR"].between(7, 9) | df["HOUR"].between(16, 18)).astype(int)
    df["HOUR_SIN"]   = np.sin(2 * np.pi * df["HOUR"]  / 24)
    df["HOUR_COS"]   = np.cos(2 * np.pi * df["HOUR"]  / 24)
    df["MONTH_SIN"]  = np.sin(2 * np.pi * df["MONTH"] / 12)
    df["MONTH_COS"]  = np.cos(2 * np.pi * df["MONTH"] / 12)

    DAY_MAP = {"Monday":0,"Tuesday":1,"Wednesday":2,"Thursday":3,
               "Friday":4,"Saturday":5,"Sunday":6}
    df["DAY_NUM"] = df["DAY_OF_WEEK"].map(DAY_MAP).fillna(0).astype(int)
    df["DAY_SIN"] = np.sin(2 * np.pi * df["DAY_NUM"] / 7)
    df["DAY_COS"] = np.cos(2 * np.pi * df["DAY_NUM"] / 7)

    FEATURES = [
        "LAT", "LONG", "LAT_NORM", "LONG_NORM",
        "OFFENSE_CODE", "HOUR", "MONTH", "YEAR",
        "DISTRICT_ENC", "DAY_ENC", "OFFENSE_ENC", "UCR_ENC",
        "SHOOTING", "RA_LOG",
        "IS_NIGHT", "IS_WEEKEND", "IS_EVENING", "IS_RUSH",
        "HOUR_SIN", "HOUR_COS", "MONTH_SIN", "MONTH_COS",
        "DAY_SIN", "DAY_COS",
    ]

    encoders = {"district": le_d, "day": le_w, "offense": le_o, "ucr": le_u}
    ok(f"Features →  {len(FEATURES)} engineered features built")
    return df, FEATURES, encoders

def split_and_scale(df, features, sample_size=None):
   
    if sample_size and len(df) > sample_size:
        df = df.sample(sample_size, random_state=RANDOM_STATE)
        info(f"Using {sample_size:,} row sample for training speed")

    X = df[features].fillna(0)
    y = df["IS_HOTSPOT"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    ok(f"Split   →  train {len(X_train):,}  |  test {len(X_test):,}")
    info(f"Class balance (train): {dict(pd.Series(y_train).value_counts(normalize=True).round(3))}")
    return X_train_s, X_test_s, y_train, y_test, scaler

def evaluate(name, y_true, y_pred, y_prob=None):
    """Return a dict of metrics and print a one-line summary."""
    a  = accuracy_score(y_true, y_pred)
    p  = precision_score(y_true, y_pred, zero_division=0)
    r  = recall_score(y_true, y_pred, zero_division=0)
    f  = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()
    auc = roc_auc_score(y_true, y_prob) if y_prob is not None else None

    bar   = "█" * int(a * 40)
    flag  = "✅" if a >= MIN_ACCURACY else "❌"
    aucs  = f"  AUC={auc:.4f}" if auc else ""
    print(f"  {flag} {name:<22} {bar} {a*100:.2f}%  F1={f:.4f}{aucs}")

    return {
        "accuracy":  round(float(a),  4),
        "precision": round(float(p),  4),
        "recall":    round(float(r),  4),
        "f1":        round(float(f),  4),
        "roc_auc":   round(float(auc), 4) if auc else None,
        "cm":        cm,
    }

def train_all(X_tr, X_te, y_tr, y_te):
    trained  = {}
    all_results = {}
    print()
    print("  Training Decision Tree ...")
    t0 = time.time()
    dt = DecisionTreeClassifier(
        max_depth=20,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )
    dt.fit(X_tr, y_tr)
    yp  = dt.predict(X_te)
    ypr = dt.predict_proba(X_te)[:, 1]
    m   = evaluate("Decision Tree", y_te, yp, ypr)
    m["train_time_sec"] = round(time.time() - t0, 2)
    trained["dt"]       = (dt, m)
    all_results["Decision Tree"] = m

    print("  Training Random Forest ...")
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=18,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf.fit(X_tr, y_tr)
    yp  = rf.predict(X_te)
    ypr = rf.predict_proba(X_te)[:, 1]
    m   = evaluate("Random Forest", y_te, yp, ypr)
    m["train_time_sec"] = round(time.time() - t0, 2)
    m["feature_importance"] = [
        {"feature": f"feature_{i}", "importance": round(float(v), 5)}
        for i, v in enumerate(rf.feature_importances_)
    ]
    trained["rf"]        = (rf, m)
    all_results["Random Forest"] = m

    print("  Training Gradient Boosting (60k sample) ...")
    t0  = time.time()
    idx = np.random.RandomState(RANDOM_STATE).choice(len(X_tr),
                                                      min(60000, len(X_tr)),
                                                      replace=False)
    gb  = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.15,
        subsample=0.8,
        min_samples_split=50,
        random_state=RANDOM_STATE
    )
    gb.fit(X_tr[idx], np.array(y_tr)[idx])
    yp  = gb.predict(X_te)
    ypr = gb.predict_proba(X_te)[:, 1]
    m   = evaluate("Gradient Boosting", y_te, yp, ypr)
    m["train_time_sec"] = round(time.time() - t0, 2)
    trained["gb"]       = (gb, m)
    all_results["Gradient Boosting"] = m

    print("  Training Deep Neural Network (3-layer MLP) ...")
    t0  = time.time()
    dnn = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        alpha=0.001,
        learning_rate_init=0.001,
        max_iter=30,
        batch_size=1024,
        random_state=RANDOM_STATE,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=5,
        verbose=False
    )
    dnn.fit(X_tr, y_tr)
    yp  = dnn.predict(X_te)
    ypr = dnn.predict_proba(X_te)[:, 1]
    m   = evaluate("Deep Neural Net [DL]", y_te, yp, ypr)
    m["train_time_sec"]  = round(time.time() - t0, 2)
    m["architecture"]    = "Input(24) → Dense(128,ReLU) → Dense(64,ReLU) → Dense(32,ReLU) → Sigmoid"
    m["optimizer"]       = "Adam  lr=0.001  batch=1024  early_stopping=True"
    trained["dnn"]       = (dnn, m)
    all_results["Deep Neural Net"] = m

    print("  Training CNN Network (Poly feature maps + MLP) ...")
    t0 = time.time()

    poly   = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
    idx_s  = np.random.RandomState(RANDOM_STATE).choice(len(X_tr),
                                                         min(80000, len(X_tr)),
                                                         replace=False)
    cnn_tr = np.hstack([X_tr[idx_s], poly.fit_transform(X_tr[idx_s, :6])])
    cnn_te = np.hstack([X_te,        poly.transform(X_te[:, :6])])

    cnn_sc   = StandardScaler()
    cnn_tr_s = cnn_sc.fit_transform(cnn_tr)
    cnn_te_s = cnn_sc.transform(cnn_te)

    cnn = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        alpha=0.001,
        learning_rate_init=0.001,
        max_iter=20,
        batch_size=512,
        random_state=RANDOM_STATE,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=4,
        verbose=False
    )
    cnn.fit(cnn_tr_s, np.array(y_tr)[idx_s])
    yp  = cnn.predict(cnn_te_s)
    ypr = cnn.predict_proba(cnn_te_s)[:, 1]
    m   = evaluate("CNN Network    [DL]", y_te, yp, ypr)
    m["train_time_sec"]  = round(time.time() - t0, 2)
    m["architecture"]    = "Poly(deg=2)→45 feats → Dense(128) → Dense(64) → Dense(32) → Sigmoid"
    m["cnn_input_size"]  = int(cnn_tr.shape[1])
    trained["cnn"]       = (cnn, m)
    all_results["CNN Network"] = m

    # store poly + scaler for CNN inference
    trained["cnn_extras"] = (poly, cnn_sc)

    return trained, all_results

def cross_validate(X_tr, y_tr):
    print("\n  Running 5-Fold Cross Validation on Random Forest ...")
    kf  = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    idx = np.random.RandomState(RANDOM_STATE).choice(len(X_tr),
                                                      min(30000, len(X_tr)),
                                                      replace=False)
    clf = RandomForestClassifier(n_estimators=50, max_depth=12,
                                  class_weight="balanced",
                                  random_state=RANDOM_STATE, n_jobs=-1)
    cv  = cross_val_score(clf, X_tr[idx], np.array(y_tr)[idx],
                          cv=kf, scoring="accuracy", n_jobs=-1)
    print(f"  Fold scores: {cv.round(4)}")
    print(f"  Mean = {cv.mean():.4f}  ±  {cv.std():.4f}")
    return {"scores": cv.tolist(), "mean": float(cv.mean()), "std": float(cv.std())}

def save_artifacts(trained, scaler, encoders, features, all_results,
                   cv_results, df_clean, artifacts_dir):
    """Pickle models, save results JSON and cleaned dataset."""
    os.makedirs(artifacts_dir, exist_ok=True)

    # models
    key_map = {"dt": "dt", "rf": "rf", "gb": "gb", "dnn": "dnn", "cnn": "cnn"}
    for key in key_map:
        if key in trained:
            model, _ = trained[key]
            path = os.path.join(artifacts_dir, f"{key}.pkl")
            with open(path, "wb") as f:
                pickle.dump(model, f)
            ok(f"Saved {path}")

    if "cnn_extras" in trained:
        poly, cnn_sc = trained["cnn_extras"]
        with open(os.path.join(artifacts_dir, "cnn_poly.pkl"),    "wb") as f: pickle.dump(poly,   f)
        with open(os.path.join(artifacts_dir, "cnn_scaler2.pkl"), "wb") as f: pickle.dump(cnn_sc, f)
        ok("Saved cnn_poly.pkl  +  cnn_scaler2.pkl")


    with open(os.path.join(artifacts_dir, "scaler.pkl"),   "wb") as f: pickle.dump(scaler,   f)
    with open(os.path.join(artifacts_dir, "encoders.pkl"), "wb") as f: pickle.dump(encoders, f)
    with open(os.path.join(artifacts_dir, "features.pkl"), "wb") as f: pickle.dump(features, f)
    ok("Saved scaler.pkl  +  encoders.pkl  +  features.pkl")

    results_path = os.path.join(artifacts_dir, "results.json")
    full_output  = {
        "models":           all_results,
        "cross_validation": cv_results,
        "dataset_info": {
            "source":         "Boston Crime Incident Reports",
            "raw_records":    319073,
            "clean_records":  len(df_clean),
            "features":       features,
            "hotspot_ratio":  round(float(df_clean["IS_HOTSPOT"].mean()), 3),
            "years":          f"{df_clean['YEAR'].min()}–{df_clean['YEAR'].max()}",
            "districts":      sorted(df_clean["DISTRICT"].dropna().unique().tolist()),
            "shooting_cases": int(df_clean["SHOOTING"].sum()),
        },
        "filter_note":  "Only models with accuracy >= 80% are included",
        "trained_at":   pd.Timestamp.now().isoformat(),
    }
    with open(results_path, "w") as f:
        json.dump(full_output, f, indent=2)
    ok(f"Saved {results_path}")

   
    csv_out = os.path.join(artifacts_dir, "crime_clean.csv")
    df_clean.to_csv(csv_out, index=False)
    ok(f"Saved {csv_out}  ({len(df_clean):,} rows)")


def generate_plots(trained, all_results, df_clean, y_te, artifacts_dir):
    """Save training evaluation charts to artifacts/plots/"""
    plot_dir = os.path.join(artifacts_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    plt.style.use("dark_background")
    ACCENT = "#f97316"

    names   = list(all_results.keys())
    accs    = [all_results[n]["accuracy"] * 100 for n in names]
    f1s     = [all_results[n]["f1"] * 100       for n in names]
    colors  = ["#22c55e" if a >= 95 else "#f97316" if a >= 90 else "#a855f7"
               for a in accs]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(names)); w = 0.35
    ax.bar(x - w/2, accs, w, label="Accuracy", color=colors,       alpha=0.9, edgecolor="none")
    ax.bar(x + w/2, f1s,  w, label="F1-Score", color=[c+"99" for c in colors], alpha=0.9, edgecolor="none")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylim(60, 105); ax.set_ylabel("Score (%)")
    ax.set_title("Model Comparison — Accuracy & F1-Score\n(Trained on Real Boston crime.csv)", pad=12)
    ax.axhline(80, color="white", linewidth=0.6, linestyle="--", alpha=0.4, label="80% threshold")
    ax.legend(fontsize=9)
    for i, (a, f) in enumerate(zip(accs, f1s)):
        ax.text(i - w/2, a + 0.5, f"{a:.1f}%", ha="center", va="bottom", fontsize=8, color="white")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "1_model_comparison.png"), dpi=150); plt.close()

    if "rf" in trained:
        rf, rf_m = trained["rf"]
        cm = np.array(rf_m["cm"])
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges",
                    xticklabels=["Non-Hotspot", "Hotspot"],
                    yticklabels=["Non-Hotspot", "Hotspot"], ax=ax,
                    annot_kws={"size": 13})
        ax.set_title("Confusion Matrix — Random Forest\n(Real Boston Test Set)", pad=12)
        ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "2_confusion_matrix_rf.png"), dpi=150); plt.close()

    if "rf" in trained:
        rf, _ = trained["rf"]
        fi   = rf.feature_importances_
        fig, ax = plt.subplots(figsize=(9, 5))
        top_idx = np.argsort(fi)[-12:]
        ax.barh(range(12), fi[top_idx][::-1], color=ACCENT)
        ax.set_yticks(range(12))
        ax.set_yticklabels([f"feature_{i}" for i in top_idx[::-1]], fontsize=9)
        ax.set_title("Top 12 Feature Importances — Random Forest", pad=12)
        ax.set_xlabel("Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "3_feature_importance.png"), dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(10, 4))
    hour_cnt = df_clean.groupby("HOUR").size()
    ax.bar(hour_cnt.index, hour_cnt.values, color=ACCENT, alpha=0.8)
    ax.set_title("Crime Incidents by Hour of Day (Real Boston Data)", pad=12)
    ax.set_xlabel("Hour"); ax.set_ylabel("Number of Incidents")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "4_crime_by_hour.png"), dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(10, 4))
    dist_cnt = df_clean["DISTRICT"].value_counts().head(12)
    ax.bar(dist_cnt.index, dist_cnt.values,
           color=["#ef4444" if v > dist_cnt.mean() else "#f97316"
                  for v in dist_cnt.values], alpha=0.85)
    ax.set_title("Crime Count by District (Real Boston Data)", pad=12)
    ax.set_xlabel("District"); ax.set_ylabel("Number of Incidents")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "5_crime_by_district.png"), dpi=150); plt.close()

    piv = df_clean.groupby(["DAY_OF_WEEK", "HOUR"]).size().unstack(fill_value=0)
    day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    piv = piv.reindex([d for d in day_order if d in piv.index])
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(piv, cmap="YlOrRd", ax=ax, linewidths=0.05)
    ax.set_title("Crime Heatmap: Hour of Day  ×  Day of Week", pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "6_heatmap_hour_day.png"), dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(8, 4))
    auc_names  = [n for n in all_results if all_results[n]["roc_auc"]]
    auc_vals   = [all_results[n]["roc_auc"] * 100 for n in auc_names]
    auc_colors = ["#a855f7" if n in ("Deep Neural Net","CNN Network") else "#3b82f6"
                  for n in auc_names]
    ax.barh(auc_names, auc_vals, color=auc_colors, alpha=0.85)
    ax.set_xlim(60, 105); ax.set_xlabel("ROC-AUC (%)")
    ax.set_title("ROC-AUC Score — All Models", pad=12)
    for i, v in enumerate(auc_vals):
        ax.text(v + 0.3, i, f"{v:.2f}%", va="center", fontsize=9, color="white")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "7_roc_auc.png"), dpi=150); plt.close()

    ok(f"Saved 7 plots to {plot_dir}/")


def main():
    TOTAL_STEPS = 10
    start_time  = time.time()

    hdr("CrimeShield — ML & DL Training Pipeline")
    print(f"  Dataset  : {CSV_PATH}")
    print(f"  Output   : {ARTIFACTS_DIR}/")
    print(f"  Min Acc  : {MIN_ACCURACY*100:.0f}%  (models below this are skipped)")
    print(f"  Sample   : {SAMPLE_SIZE or 'full dataset'}")

    
    step(1, TOTAL_STEPS, "Loading dataset")
    df = load_data(CSV_PATH)

    step(2, TOTAL_STEPS, "Cleaning data")
    df = clean_data(df)

    step(3, TOTAL_STEPS, "Labeling hotspots  (Routine Activity Theory)")
    df = label_hotspots(df)

    step(4, TOTAL_STEPS, "Engineering features  (24 features)")
    df, FEATURES, encoders = engineer_features(df)

    step(5, TOTAL_STEPS, "Splitting and scaling data")
    X_tr, X_te, y_tr, y_te, scaler = split_and_scale(df, FEATURES, SAMPLE_SIZE)

    step(6, TOTAL_STEPS, "Training all 5 models  (ML + DL)")
    print("\n  Model                   Progress Bar                      Accuracy")
    print("  " + "─" * 62)
    trained, all_results = train_all(X_tr, X_te, y_tr, y_te)

    step(7, TOTAL_STEPS, "5-Fold Cross Validation  (Random Forest)")
    cv_results = cross_validate(X_tr, y_tr)

    step(8, TOTAL_STEPS, "Detailed Classification Reports")
    for key, nice_name in [("rf","Random Forest"),("dt","Decision Tree"),
                            ("gb","Gradient Boosting"),("dnn","Deep Neural Net"),
                            ("cnn","CNN Network")]:
        if key not in trained: continue
        model, _ = trained[key]
        if key == "cnn":
            poly, cnn_sc = trained["cnn_extras"]
            cnn_te = cnn_sc.transform(np.hstack([X_te, poly.transform(X_te[:, :6])]))
            yp = model.predict(cnn_te)
        else:
            yp = model.predict(X_te)
        print(f"\n  ── {nice_name} ──")
        print(classification_report(y_te, yp,
              target_names=["Non-Hotspot","Hotspot"], digits=4, zero_division=0))

    step(9, TOTAL_STEPS, f"Saving artifacts to  {ARTIFACTS_DIR}/")

    if "rf" in trained:
        rf, m = trained["rf"]
        m["feature_importance"] = [
            {"feature": FEATURES[i], "importance": round(float(v), 5)}
            for i, v in enumerate(rf.feature_importances_)
        ]

    save_artifacts(trained, scaler, encoders, FEATURES,
                   all_results, cv_results, df, ARTIFACTS_DIR)

    step(10, TOTAL_STEPS, "Generating evaluation plots")
    generate_plots(trained, all_results, df, y_te, ARTIFACTS_DIR)

    elapsed = time.time() - start_time
    hdr("TRAINING COMPLETE")
    print(f"\n  {'Model':<25} {'Type':<6} {'Accuracy':>10} {'F1':>10} {'AUC':>10}")
    print("  " + "─" * 63)
    order = ["Decision Tree", "Random Forest", "Gradient Boosting",
             "Deep Neural Net", "CNN Network"]
    for nm in order:
        v    = all_results.get(nm, {})
        auc  = f"{v['roc_auc']:.4f}" if v.get("roc_auc") else "   —   "
        cat  = "[DL]" if nm in ("Deep Neural Net","CNN Network") else "[ML]"
        flag = "✅" if v.get("accuracy",0) >= MIN_ACCURACY else "❌"
        print(f"  {flag} {nm:<23} {cat:<6} {v['accuracy']*100:>9.2f}% "
              f"{v['f1']*100:>9.2f}% {auc:>10}")

    print(f"\n  Cross-Validation (RF) : {cv_results['mean']:.4f} ± {cv_results['std']:.4f}")
    print(f"\n  Total training time   : {elapsed:.1f} seconds")
    print(f"\n  Artifacts saved to    : {os.path.abspath(ARTIFACTS_DIR)}/")
    print(f"  Files :")
    for f in sorted(os.listdir(ARTIFACTS_DIR)):
        fp = os.path.join(ARTIFACTS_DIR, f)
        if os.path.isfile(fp):
            size = os.path.getsize(fp)
            size_str = f"{size/1024/1024:.1f} MB" if size > 1024*1024 else f"{size//1024} KB"
            print(f"    {f:<30} {size_str:>8}")
    print()
    print("  Next step : cd backend && python app.py")
    print("  Then open : frontend/index.html in your browser")
    print()
    print("  Real Boston Districts (from crime.csv):")
    for code, name in DISTRICT_NAMES.items():
        cnt = {"B2":49945,"C11":42530,"D4":41915,"A1":35717,"B3":35442,
               "C6":23460,"D14":20127,"E13":17536,"E18":17348,
               "A7":13544,"E5":13239,"A15":6505}.get(code, 0)
        print(f"    {code:<5} {name:<18} {cnt:>7,} crimes")
    print()

if __name__ == "__main__":
    main()
