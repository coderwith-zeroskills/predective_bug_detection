import numpy as np
import scipy.sparse as sp
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, recall_score, precision_score, f1_score
)
from xgboost import XGBClassifier

# ── 1. Load features ───────────────────────────────────────────────────────────

X = sp.load_npz("features.npz")
y = np.load("labels.npy")

print(f"Features loaded: {X.shape}")
print(f"Labels loaded  : {y.shape}  ({y.sum()} high-risk, {(y==0).sum()} safe)\n")

# ── 2. Train / test split ──────────────────────────────────────────────────────
# 80% train, 20% test
# stratify=y ensures both splits have balanced 50/50 labels

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set : {X_train.shape[0]} stories")
print(f"Test set     : {X_test.shape[0]} stories\n")

# ── 3. Train XGBoost ───────────────────────────────────────────────────────────
# scale_pos_weight: tells XGBoost to penalise missing a positive (risky) story
# more than making a false alarm. This is how we push recall higher.
# Value = count(safe) / count(risky) = 250/250 = 1 here, but we set it to 2
# to deliberately bias toward catching all risky stories.

model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    scale_pos_weight=2,       # bias toward catching high-risk stories
    # use_label_encoder=False,
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train, y_train)
print("Model trained.\n")

# ── 4. Predict ─────────────────────────────────────────────────────────────────
# predict_proba gives a probability score (0.0 to 1.0)
# We lower the threshold from default 0.5 to 0.35
# meaning: "flag as risky if there's even a 35% chance"
# This increases recall (catch more risky) at cost of some precision

y_prob  = model.predict_proba(X_test)[:, 1]
threshold = 0.35
y_pred  = (y_prob >= threshold).astype(int)

# ── 5. Metrics ─────────────────────────────────────────────────────────────────

recall    = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
roc_auc   = roc_auc_score(y_test, y_prob)

print("=" * 45)
print("  MODEL PERFORMANCE")
print("=" * 45)
print(f"  Recall    : {recall:.2%}   ← % of risky stories caught")
print(f"  Precision : {precision:.2%}   ← % of flagged that are truly risky")
print(f"  F1 Score  : {f1:.2%}   ← balance of both")
print(f"  ROC-AUC   : {roc_auc:.2%}   ← overall discrimination ability")
print("=" * 45)

print("\nDetailed breakdown:")
print(classification_report(y_test, y_pred, target_names=["Safe", "High Risk"]))

# ── 6. Confusion matrix ────────────────────────────────────────────────────────

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print("Confusion Matrix:")
print(f"  True Negatives  (correctly called safe)  : {tn}")
print(f"  False Positives (safe flagged as risky)   : {fp}  ← false alarms")
print(f"  False Negatives (risky missed entirely)   : {fn}  ← the dangerous ones")
print(f"  True Positives  (correctly caught risky)  : {tp}")

# ── 7. Cross-validation ────────────────────────────────────────────────────────
# Tests the model on 5 different splits to confirm results aren't a fluke

print("\nRunning 5-fold cross-validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=cv, scoring="recall")
print(f"  Recall per fold : {[f'{s:.2%}' for s in cv_scores]}")
print(f"  Mean recall     : {cv_scores.mean():.2%} (+/- {cv_scores.std():.2%})")

# ── 8. Top predictive features ─────────────────────────────────────────────────

print("\nTop 15 features driving HIGH RISK predictions:")
feature_names = (
    [f"tfidf_{i}" for i in range(500)] +
    ["story_points", "num_comments"] +
    ["kw_auth","kw_payment","kw_migrat","kw_async","kw_webhook",
     "kw_encrypt","kw_oauth","kw_schema","kw_legacy","kw_permiss",
     "kw_refactor","kw_fraud","kw_billing","kw_compliance","kw_race"]
)

importances = model.feature_importances_
top_idx = np.argsort(importances)[::-1][:15]
for rank, idx in enumerate(top_idx, 1):
    print(f"  {rank:2}. {feature_names[idx]:<25} importance: {importances[idx]:.4f}")

# ── 9. Save model ──────────────────────────────────────────────────────────────

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("threshold.pkl", "wb") as f:
    pickle.dump(threshold, f)

print("\nSaved: model.pkl, threshold.pkl")

# ── 10. Quick chart ────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Confusion matrix heatmap
ax = axes[0]
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(["Pred Safe", "Pred Risky"])
ax.set_yticklabels(["Actual Safe", "Actual Risky"])
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                fontsize=18, color="white" if cm[i, j] > cm.max()/2 else "black")
ax.set_title("Confusion Matrix")

# Feature importance bar chart
ax2 = axes[1]
top_names  = [feature_names[i] for i in top_idx[:10]]
top_scores = [importances[i] for i in top_idx[:10]]
ax2.barh(top_names[::-1], top_scores[::-1], color="#4f83cc")
ax2.set_title("Top 10 Features")
ax2.set_xlabel("Importance")

plt.tight_layout()
plt.savefig("model_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("Chart saved: model_results.png")
print("\nPhase 4 complete.")