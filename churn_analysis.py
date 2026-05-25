import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import os

os.makedirs("charts", exist_ok=True)

# ── 1. Load Data ───────────────────────────────────────────────────────────────
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
print(f"Dataset shape: {df.shape}")
print(df.head())

# ── 2. Clean Data ──────────────────────────────────────────────────────────────
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
print("\n✅ Data cleaned")

# ── 3. Chart 1 — Churn Distribution ───────────────────────────────────────────
churn_counts = df["Churn"].value_counts()
colors = ["#4CAF82", "#E05C5C"]
plt.figure(figsize=(6, 4))
plt.bar(["No Churn", "Churned"], churn_counts.values, color=colors, width=0.4)
plt.title("Customer Churn Distribution", fontsize=13, fontweight="bold", pad=15)
plt.ylabel("Number of Customers", fontsize=11)
plt.xticks(fontsize=11)
for i, v in enumerate(churn_counts.values):
    plt.text(i, v + 30, str(v), ha="center", fontsize=10)
plt.gca().spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("charts/01_churn_distribution.png", dpi=150)
plt.close()
print("✅ Chart 1 saved")

# ── 4. Chart 2 — Monthly Charges by Churn ─────────────────────────────────────
churned = df[df["Churn"] == 1]["MonthlyCharges"]
not_churned = df[df["Churn"] == 0]["MonthlyCharges"]
plt.figure(figsize=(8, 4))
plt.hist(not_churned, bins=30, alpha=0.6, color="#4CAF82", label="No Churn")
plt.hist(churned, bins=30, alpha=0.6, color="#E05C5C", label="Churned")
plt.title("Monthly Charges — Churned vs Not Churned", fontsize=13, fontweight="bold", pad=15)
plt.xlabel("Monthly Charges ($)", fontsize=11)
plt.ylabel("Number of Customers", fontsize=11)
plt.legend(fontsize=10)
plt.gca().spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("charts/02_monthly_charges.png", dpi=150)
plt.close()
print("✅ Chart 2 saved")

# ── 5. Chart 3 — Tenure by Churn ──────────────────────────────────────────────
plt.figure(figsize=(8, 4))
plt.hist(df[df["Churn"] == 0]["tenure"], bins=30, alpha=0.6, color="#4CAF82", label="No Churn")
plt.hist(df[df["Churn"] == 1]["tenure"], bins=30, alpha=0.6, color="#E05C5C", label="Churned")
plt.title("Customer Tenure — Churned vs Not Churned", fontsize=13, fontweight="bold", pad=15)
plt.xlabel("Tenure (Months)", fontsize=11)
plt.ylabel("Number of Customers", fontsize=11)
plt.legend(fontsize=10)
plt.gca().spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("charts/03_tenure_distribution.png", dpi=150)
plt.close()
print("✅ Chart 3 saved")

# ── 6. Train Model ─────────────────────────────────────────────────────────────
features = ["tenure", "MonthlyCharges", "TotalCharges"]
X = df[features]
y = df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"\n── Model Results ──")
print(f"Features used:  {features}")
print(f"Training size:  {len(X_train)} rows")
print(f"Testing size:   {len(X_test)} rows")
print(f"Model Accuracy: {round(accuracy * 100, 2)}%")

# ── 7. Chart 4 — Feature Importance ───────────────────────────────────────────
importance = dict(zip(features, model.coef_[0]))
plt.figure(figsize=(6, 4))
bars = plt.barh(list(importance.keys()), list(importance.values()),
                color=["#4A90D9", "#E05C5C", "#F5A623"])
plt.title("Feature Importance (Logistic Regression Coefficients)",
          fontsize=11, fontweight="bold", pad=15)
plt.xlabel("Coefficient Value", fontsize=10)
plt.axvline(0, color="black", linewidth=0.8)
plt.gca().spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("charts/04_feature_importance.png", dpi=150)
plt.close()
print("✅ Chart 4 saved")

print("\n✅ All done! Files created:")
print("   charts/01_churn_distribution.png")
print("   charts/02_monthly_charges.png")
print("   charts/03_tenure_distribution.png")
print("   charts/04_feature_importance.png")