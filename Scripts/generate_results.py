"""
Generate analytical result artifacts for the UK Traffic Accidents project.

Reads the SQLite database (2020 accidents) and writes to ../Results:
  - association_rules.csv / .md   (Apriori association-rule mining)
  - cluster_assignments.csv       (regional KMeans cluster per accident)
  - cluster_centroids.csv         (cluster centres + sizes)
  - model_metrics.csv / .json     (fatal-injury classifier evaluation)
  - confusion_matrix.png          (classifier confusion matrix)

Implemented with numpy / pandas only (no scikit-learn / mlxtend required).
Set the database path via the ACC_DB environment variable, or place the
database at ../Data/accident_data_v1.0.0_2023.db.
"""
import sqlite3, os, json, itertools
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES  = os.path.join(ROOT, "Results"); os.makedirs(RES, exist_ok=True)
DB   = os.environ.get("ACC_DB", os.path.join(ROOT, "Data", "accident_data_v1.0.0_2023.db"))

con = sqlite3.connect(DB)
acc = pd.read_sql_query("SELECT * FROM accident WHERE accident_year=2020", con)

sev={1:"Fatal",2:"Serious",3:"Slight"}
urb={1:"Urban",2:"Rural",3:"Unallocated"}
light={1:"Daylight",4:"Dark-lit",5:"Dark-unlit",6:"Dark-no-lighting",7:"Dark-unknown"}
weather={1:"Fine",2:"Raining",3:"Snowing",4:"Fine+winds",5:"Raining+winds",6:"Snowing+winds",7:"Fog",8:"Other",9:"Unknown"}
surf={1:"Dry",2:"Wet",3:"Snow",4:"Frost/ice",5:"Flood",6:"Oil",7:"Mud"}

# =========================================================
# 1. APRIORI ASSOCIATION RULES  (vectorized one-hot support)
# =========================================================
tx = pd.DataFrame({
    "severity":    acc["accident_severity"].map(sev),
    "urban_rural": acc["urban_or_rural_area"].map(urb),
    "light":       acc["light_conditions"].map(light),
    "weather":     acc["weather_conditions"].map(weather),
    "surface":     acc["road_surface_conditions"].map(surf),
    "speed":       acc["speed_limit"].astype("Int64").astype(str) + "mph",
}).dropna()

MIN_SUP = 0.05
oh_df = pd.get_dummies(tx, prefix_sep="=")
oh_df = oh_df.loc[:, oh_df.mean() >= MIN_SUP]
cols  = list(oh_df.columns)
oh    = oh_df.to_numpy(bool)
name  = {i: c for i, c in enumerate(cols)}

def sup_idx(idxs):
    return oh[:, list(idxs)].all(axis=1).mean()

freq = {}
singles = []
for i in range(len(cols)):
    sv = oh[:, i].mean()
    if sv >= MIN_SUP:
        freq[frozenset([i])] = sv; singles.append(i)

pairs = []
for a, b in itertools.combinations(singles, 2):
    sv = sup_idx((a, b))
    if sv >= MIN_SUP:
        freq[frozenset([a, b])] = sv; pairs.append((a, b))

triples = set()
for (a, b) in pairs:
    for s in singles:
        if s in (a, b): continue
        triples.add(frozenset([a, b, s]))
for cand in triples:
    if all(frozenset(sp) in freq for sp in itertools.combinations(cand, 2)):
        sv = sup_idx(tuple(cand))
        if sv >= MIN_SUP:
            freq[cand] = sv

rules = []
for iset, sup in list(freq.items()):
    if len(iset) < 2: continue
    for r in range(1, len(iset)):
        for ante in itertools.combinations(iset, r):
            ante = frozenset(ante); cons = iset - ante
            sa = freq.get(ante) or sup_idx(tuple(ante))
            sc = freq.get(cons) or sup_idx(tuple(cons))
            if sa <= 0: continue
            conf = sup / sa; lift = conf / sc if sc > 0 else np.nan
            rules.append({
                "antecedent":  ", ".join(sorted(name[i] for i in ante)),
                "consequent":  ", ".join(sorted(name[i] for i in cons)),
                "support":     round(sup, 3),
                "confidence":  round(conf, 3),
                "lift":        round(lift, 3),
            })
rules_df = (pd.DataFrame(rules)
            .query("lift > 1")
            .sort_values(["lift", "confidence"], ascending=False)
            .head(25).reset_index(drop=True))
rules_df.to_csv(os.path.join(RES, "association_rules.csv"), index=False)
open(os.path.join(RES, "association_rules.md"), "w").write(
    "### Top Association Rules (Apriori, min support 0.05, lift > 1)\n\n"
    + rules_df.to_markdown(index=False) + "\n")
print("association rules:", len(rules_df))

# =========================================================
# 2. KMeans clustering (numpy) — Humberside region
# =========================================================
reg = pd.read_sql_query(
    "SELECT accident_index, longitude, latitude FROM accident "
    "WHERE police_force=16 AND accident_year>=2020", con).dropna()
reg = reg[(reg.longitude.between(-1.5, 0.5)) & (reg.latitude.between(53.3, 54.3))]

def kmeans(X, k, iters=100, seed=42):
    rng = np.random.default_rng(seed)
    cen = X[rng.choice(len(X), k, replace=False)]
    for _ in range(iters):
        lab = ((X[:, None, :] - cen[None, :, :]) ** 2).sum(2).argmin(1)
        new = np.array([X[lab == j].mean(0) if (lab == j).any() else cen[j] for j in range(k)])
        if np.allclose(new, cen): break
        cen = new
    return lab, cen

X = reg[["longitude", "latitude"]].to_numpy()
lab, cen = kmeans(X, 5)
reg = reg.assign(cluster=lab + 1)
reg.to_csv(os.path.join(RES, "cluster_assignments.csv"), index=False)
cc = pd.DataFrame(cen, columns=["longitude", "latitude"]); cc.insert(0, "cluster", range(1, 6))
cc["n_accidents"] = [int((lab == k).sum()) for k in range(5)]
cc.to_csv(os.path.join(RES, "cluster_centroids.csv"), index=False)
print("clusters written")

# =========================================================
# 3. Fatal-injury classifier (numpy logistic regression)
# =========================================================
# Leakage-free cross-check: predict fatality from pre-outcome features only
# (deliberately excludes casualty_severity, which defines the target).
feat = ["speed_limit", "number_of_vehicles", "number_of_casualties", "urban_or_rural_area",
        "light_conditions", "road_surface_conditions", "road_type", "day_of_week"]
d = acc[feat + ["accident_severity"]].copy()
for c in feat:
    d[c] = pd.to_numeric(d[c], errors="coerce")
d = d.dropna()
d["fatal"] = (d["accident_severity"] == 1).astype(int)

pos = d[d.fatal == 1]
neg = d[d.fatal == 0].sample(len(pos), random_state=42)
bal = pd.concat([pos, neg]).sample(frac=1, random_state=42)

Xf = bal[feat].to_numpy(float); y = bal["fatal"].to_numpy(float)
Xf = (Xf - Xf.mean(0)) / Xf.std(0)
Xf = np.c_[np.ones(len(Xf)), Xf]
rng = np.random.default_rng(0); idx = rng.permutation(len(Xf)); cut = int(0.8 * len(Xf))
tr, te = idx[:cut], idx[cut:]

w = np.zeros(Xf.shape[1]); lr = 0.1
for _ in range(3000):
    p = 1 / (1 + np.exp(-Xf[tr] @ w))
    w -= lr * Xf[tr].T @ (p - y[tr]) / len(tr)

pred = (1 / (1 + np.exp(-Xf[te] @ w)) >= 0.5).astype(int); yt = y[te].astype(int)
tp = int(((pred == 1) & (yt == 1)).sum()); tn = int(((pred == 0) & (yt == 0)).sum())
fp = int(((pred == 1) & (yt == 0)).sum()); fn = int(((pred == 0) & (yt == 1)).sum())
acc_ = (tp + tn) / len(yt)
prec = tp / (tp + fp) if tp + fp else 0
rec  = tp / (tp + fn) if tp + fn else 0
f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0
metrics = {"model": "Logistic Regression (balanced, numpy)",
           "target": "fatal accident (severity = Fatal)",
           "n_train": int(len(tr)), "n_test": int(len(te)),
           "accuracy": round(acc_, 3), "precision": round(prec, 3),
           "recall": round(rec, 3), "f1": round(f1, 3),
           "confusion_matrix": {"TN": tn, "FP": fp, "FN": fn, "TP": tp}}
json.dump(metrics, open(os.path.join(RES, "model_metrics.json"), "w"), indent=2)
pd.DataFrame([{"metric": k, "value": metrics[k]} for k in ["accuracy", "precision", "recall", "f1"]]
             ).to_csv(os.path.join(RES, "model_metrics.csv"), index=False)

fig, ax = plt.subplots(figsize=(5, 4))
cm = np.array([[tn, fp], [fn, tp]])
ax.imshow(cm, cmap="Blues")
for (i, j), v in np.ndenumerate(cm):
    ax.text(j, i, str(v), ha="center", va="center",
            color="white" if v > cm.max() / 2 else "black", fontsize=13)
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(["Non-fatal", "Fatal"]); ax.set_yticklabels(["Non-fatal", "Fatal"])
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title("Fatal-Injury Classifier — Confusion Matrix")
fig.savefig(os.path.join(RES, "confusion_matrix.png"), bbox_inches="tight"); plt.close(fig)
print("model metrics:", metrics)
