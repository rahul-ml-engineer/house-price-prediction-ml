import joblib
import pandas as pd
import matplotlib.pyplot as plt

from config import MODEL_PATH

# Load trained pipeline
pipeline = joblib.load(MODEL_PATH)

model = pipeline.named_steps["model"]
preprocessor = pipeline.named_steps["preprocessor"]

# Get feature names after preprocessing
feature_names = preprocessor.get_feature_names_out()

# Get feature importance
importances = model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

# Sort by importance
importance_df = importance_df.sort_values("Importance", ascending=False)

print("\nTop 15 Important Features:\n")
print(importance_df.head(15))

# Plot feature importance
top_features = importance_df.head(15)

plt.figure(figsize=(10,6))
plt.barh(top_features["Feature"], top_features["Importance"])
plt.gca().invert_yaxis()

plt.title("Top Features Affecting House Price")
plt.xlabel("Importance")

plt.tight_layout()
plt.savefig("reports/figures/feature_importance.png")
plt.show()