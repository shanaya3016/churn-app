import json, joblib, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
                             RocCurveDisplay, classification_report)
from utils import load_clean, engineer

# ---------- 1. DATA PROCESSING ----------
df = load_clean()
print("Shape:", df.shape)
print("Missing values:\n", df.isna().sum()[df.isna().sum() > 0])   # TotalCharges NaN -> imputer handle karega

# Outlier detection (IQR) + capping
for c in ["tenure", "MonthlyCharges", "TotalCharges"]:
    q1, q3 = df[c].quantile([.25, .75]); iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    print(f"{c}: {((df[c] < lo) | (df[c] > hi)).sum()} outliers")
    df[c] = df[c].clip(lo, hi)

plt.figure(figsize=(8, 3))
sns.boxplot(data=df[["MonthlyCharges", "tenure"]], orient="h")
plt.title("Outlier check"); plt.tight_layout(); plt.savefig("assets/outliers.png"); plt.close()

y = (df["Churn"] == "Yes").astype(int)
X = engineer(df.drop(columns="Churn"))                             # feature engineering

num_cols = X.select_dtypes("number").columns.tolist()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()

pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), num_cols),        # scaling
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),  # encoding
])

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# ---------- 2. MODEL DEVELOPMENT (4 models) ----------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "KNN": KNeighborsClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=3,
                                            class_weight="balanced", random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}

def evaluate(pipe):
    pred = pipe.predict(X_te); proba = pipe.predict_proba(X_te)[:, 1]
    return {"Accuracy": accuracy_score(y_te, pred), "Precision": precision_score(y_te, pred),
            "Recall": recall_score(y_te, pred), "F1": f1_score(y_te, pred),
            "ROC-AUC": roc_auc_score(y_te, proba)}

slug = lambda n: n.lower().replace(" ", "_")
fitted, rows = {}, []
for name, m in models.items():
    pipe = Pipeline([("pre", pre), ("model", m)]).fit(X_tr, y_tr)
    fitted[name] = pipe
    rows.append({"Model": name, "Stage": "Before tuning", **evaluate(pipe)})
    joblib.dump(pipe, f"models/{slug(name)}.joblib", compress=3)

before = pd.DataFrame(rows)
print(before.round(4))
best_name = before.sort_values("ROC-AUC", ascending=False).iloc[0]["Model"]
print("Best model:", best_name)

# ---------- 3. HYPERPARAMETER TUNING ----------
grids = {
    "Logistic Regression": {"model__C": [0.01, 0.1, 1, 10]},
    "KNN": {"model__n_neighbors": [5, 9, 15, 25], "model__weights": ["uniform", "distance"]},
    "Random Forest": {"model__n_estimators": [150, 250], "model__max_depth": [8, 12, 16],
                      "model__min_samples_leaf": [2, 4]},
    "Gradient Boosting": {"model__n_estimators": [100, 200], "model__learning_rate": [0.03, 0.05, 0.1],
                          "model__max_depth": [2, 3]},
}
gs = GridSearchCV(fitted[best_name], grids[best_name], cv=5, scoring="roc_auc", n_jobs=-1)
gs.fit(X_tr, y_tr)
tuned = gs.best_estimator_
print("Best params:", gs.best_params_)

after = pd.DataFrame([{"Model": best_name, "Stage": "After tuning", **evaluate(tuned)}])
results = pd.concat([before, after], ignore_index=True)
results.to_csv("models/results.csv", index=False)
print(results.round(4))

# ---------- 4. SAVE FINAL MODEL ----------
joblib.dump(tuned, "models/best_model.joblib", compress=3)
json.dump({"best_model": best_name, "best_params": gs.best_params_}, open("models/meta.json", "w"))

# ---------- Plots + importance ----------
print(classification_report(y_te, tuned.predict(X_te)))
ConfusionMatrixDisplay(confusion_matrix(y_te, tuned.predict(X_te)),
                       display_labels=["Stay", "Churn"]).plot(cmap="Blues")
plt.title("Confusion Matrix (tuned)"); plt.savefig("assets/confusion_matrix.png", bbox_inches="tight"); plt.close()

fig, ax = plt.subplots(figsize=(6, 5))
for name, p in fitted.items():
    RocCurveDisplay.from_estimator(p, X_te, y_te, ax=ax, name=name)
ax.plot([0, 1], [0, 1], "k--"); plt.title("ROC Curves")
plt.savefig("assets/roc_curves.png", bbox_inches="tight"); plt.close()

imp = permutation_importance(tuned, X_te, y_te, n_repeats=5, random_state=42, scoring="roc_auc")
pd.DataFrame({"feature": X_te.columns, "importance": imp.importances_mean}) \
  .sort_values("importance", ascending=False).to_csv("models/feature_importance.csv", index=False)
print("Done. Models saved.")