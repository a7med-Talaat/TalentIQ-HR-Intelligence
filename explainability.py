import json
import os
import pickle
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import shap
from train_model import _IsotonicCalibratedModel

# --- Step 1: Load saved models and settings ---
print("Loading model and files...")

with open("Trained_Model/preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

with open("Trained_Model/feature_pipeline.pkl", "rb") as f:
    feature_pipeline = pickle.load(f)

with open("Trained_Model/best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("Trained_Model/selected_features.json", "r") as f:
    selected_features = json.load(f)

# Make sure output folders exist
os.makedirs("outputs/shap_plots", exist_ok=True)


# --- Step 2: Load and prepare test data ---
print("Reading data...")
test_path = Path("Datasets/1_Raw_Data/aug_test.csv")
if not test_path.exists():
    test_path = Path("Datasets/aug_test.csv")

data = pd.read_csv(test_path)
sample_data = data.head(5)

# Transform data using saved pipelines
transformed = preprocessor.transform(sample_data)
transformed = feature_pipeline.transform(transformed)

# Keep only the features used during training
kept_features = []
for col in selected_features:
    if col in transformed.columns:
        kept_features.append(col)

X = transformed[kept_features].values


# --- Step 3: Set up SHAP Explainer ---
# If wrapped in a calibrator, get the inner tree model
base_model = getattr(model, "base_model", model)
explainer = shap.TreeExplainer(base_model)
shap_values = explainer.shap_values(X)

# Handle multi-class / binary list outputs from SHAP
if isinstance(shap_values, list):
    shap_values = shap_values[1]

# Get the baseline value
if isinstance(explainer.expected_value, list):
    base_value = float(explainer.expected_value[1])
else:
    base_value = float(explainer.expected_value)


# --- Step 4: Explain each candidate one by one ---
all_results = []

for i in range(len(sample_data)):
    row_values = X[i]
    row_shap = shap_values[i]

    # Predict probability of class 1
    prob = model.predict_proba(row_values.reshape(1, -1))[0][1]

    # Candidate ID
    if "enrollee_id" in sample_data.columns:
        candidate_id = int(sample_data["enrollee_id"].iloc[i])
    else:
        candidate_id = i

    # Collect individual feature impacts
    feature_impacts = []
    for j in range(len(kept_features)):
        feature_name = kept_features[j]
        actual_val = round(float(row_values[j]), 4)
        impact = round(float(row_shap[j]), 6)

        feature_impacts.append({
            "feature": feature_name,
            "value": actual_val,
            "shap": impact,
        })

    # Sort so the highest absolute impact is at the top
    feature_impacts.sort(key=lambda item: abs(item["shap"]), reverse=True)

    # Save waterfall plot
    plot_file = f"outputs/shap_plots/candidate_{candidate_id}_shap.png"
    explanation_obj = shap.Explanation(
        values=row_shap,
        base_values=base_value,
        data=row_values,
        feature_names=kept_features,
    )
    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(explanation_obj, max_display=15, show=False)
    plt.savefig(plot_file, bbox_inches="tight")
    plt.close()

    # Save dictionary for this applicant
    applicant_data = {
        "candidate_id": candidate_id,
        "prediction": int(prob >= 0.5),
        "probability": round(float(prob), 4),
        "base_value": round(base_value, 6),
        "shap_values": feature_impacts,
        "plot_path": plot_file,
    }
    all_results.append(applicant_data)


# --- Step 5: Save results to JSON and display ---
with open("outputs/batch_explanations.json", "w") as f:
    json.dump(all_results, f, indent=2)

print(f"Computed explanations for {len(all_results)} applicants.")
top_feature = all_results[0]["shap_values"][0]
print(f"Top driver for candidate #{all_results[0]['candidate_id']}: {top_feature}")
