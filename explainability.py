import json
import pickle
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')
from train_model import _IsotonicCalibratedModel
MODELS_DIR = Path('Trained_Model')
OUTPUTS_DIR = Path('outputs')
PLOTS_DIR = OUTPUTS_DIR / 'shap_plots'

def load_artifacts(models_dir=MODELS_DIR):

    def _pkl(name):
        path = models_dir / name
        if not path.exists():
            raise FileNotFoundError(f'Missing: {path} — run train_model.py first.')
        with open(path, 'rb') as f:
            return pickle.load(f)
    with open(models_dir / 'selected_features.json') as f:
        features = json.load(f)
    with open(models_dir / 'model_metrics.json') as f:
        metadata = json.load(f)
    return {'preprocessor': _pkl('preprocessor.pkl'), 'feature_pipeline': _pkl('feature_pipeline.pkl'), 'best_model': _pkl('best_model.pkl'), 'selected_features': features, 'best_model_name': metadata.get('best_model', 'Calibrated Classifier')}

def _transform_to_features(df, artifacts):
    X = artifacts['preprocessor'].transform(df)
    X = artifacts['feature_pipeline'].transform(X)
    feature_names = [c for c in artifacts['selected_features'] if c in X.columns]
    return (X[feature_names].values, feature_names)

def _build_shap_explainer(model, X_background):
    import shap
    inner = getattr(model, 'base_model', model)
    model_type = type(inner).__name__.lower()
    n_bg = min(50, X_background.shape[0])
    if 'lgbm' in model_type or 'xgb' in model_type:
        return shap.TreeExplainer(inner)
    elif 'logistic' in model_type:
        return shap.LinearExplainer(inner, shap.kmeans(X_background, n_bg))
    else:
        return shap.KernelExplainer(model.predict_proba, shap.kmeans(X_background, n_bg))

def explain_candidate(candidate_id, X_row, feature_names, model, explainer, output_dir=PLOTS_DIR, save_plot=True):
    import shap, matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output_dir.mkdir(parents=True, exist_ok=True)
    prob = float(model.predict_proba(X_row.reshape(1, -1))[0, 1])
    sv = explainer.shap_values(X_row.reshape(1, -1))
    if isinstance(sv, list):
        shap_vals = sv[1][0] if len(sv) > 1 else sv[0][0]
    else:
        shap_vals = sv[0] if sv.ndim == 2 else sv
    if isinstance(explainer.expected_value, (list, np.ndarray)):
        base_value = float(explainer.expected_value[1])
    else:
        base_value = float(explainer.expected_value)

    contributions = []
    for i in range(len(feature_names)):
        feat = feature_names[i]
        val = round(float(X_row[i]), 4)
        shap_score = round(float(shap_vals[i]), 6)
        contributions.append({'feature': feat, 'value': val, 'shap': shap_score})
    contributions.sort(key=lambda item: abs(item['shap']), reverse=True)

    if hasattr(candidate_id, 'item'):
        c_id = int(candidate_id)
    else:
        c_id = candidate_id

    result = {
        'candidate_id': c_id,
        'prediction': int(prob >= 0.5),
        'probability': round(prob, 4),
        'base_value': round(base_value, 6),
        'shap_values': contributions
    }
    if save_plot:
        try:
            expl_obj = shap.Explanation(values=shap_vals, base_values=base_value, data=X_row, feature_names=feature_names)
            fig, _ = plt.subplots(figsize=(10, 6))
            shap.waterfall_plot(expl_obj, max_display=15, show=False)
            plot_path = output_dir / f'candidate_{candidate_id}_shap.png'
            plt.savefig(plot_path, bbox_inches='tight', dpi=120)
            plt.close()
            result['plot_path'] = str(plot_path)
        except Exception:
            pass
    return result

def explain_batch(df, artifacts, output_dir=PLOTS_DIR, id_col='enrollee_id', save_json=True, save_summary=True):
    import shap, matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output_dir.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    model = artifacts['best_model']
    X_all, features = _transform_to_features(df, artifacts)
    explainer = _build_shap_explainer(model, X_all)
    if save_summary:
        try:
            sv_all = explainer.shap_values(X_all)
            sv_matrix = sv_all[1] if isinstance(sv_all, list) and len(sv_all) > 1 else sv_all[0] if isinstance(sv_all, list) else sv_all
            fig, _ = plt.subplots(figsize=(10, 8))
            shap.summary_plot(sv_matrix, X_all, feature_names=features, show=False)
            plt.savefig(OUTPUTS_DIR / 'shap_summary.png', bbox_inches='tight', dpi=120)
            plt.close()
        except Exception:
            pass
    ids = df[id_col].values if id_col in df.columns else range(len(df))
    results = [explain_candidate(cid, X_all[i], features, model, explainer, output_dir) for i, cid in enumerate(ids)]
    if save_json:
        with open(OUTPUTS_DIR / 'batch_explanations.json', 'w') as f:
            json.dump(results, f, indent=2)
    return results
if __name__ == '__main__':
    arts = load_artifacts()
    csv_path = Path('Datasets/1_Raw_Data/aug_test.csv')
    if not csv_path.exists():
        csv_path = Path('Datasets/aug_test.csv')
    test_df = pd.read_csv(csv_path).head(5)
    exps = explain_batch(test_df, arts, save_summary=False)
    print(f"Computed explanations for {len(exps)} applicants. Top driver for candidate #{exps[0]['candidate_id']}: {exps[0]['shap_values'][0]}")
