import pickle
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE, VarianceThreshold, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

def compute_vif(X, threshold=10.0):
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        X_num = X.select_dtypes(include=[np.number]).dropna(axis=1)
        rows = []
        for i, col in enumerate(X_num.columns):
            try:
                vif = variance_inflation_factor(X_num.values, i)
            except Exception:
                vif = float('nan')
            rows.append({'feature': col, 'VIF': round(vif, 3)})
        df = pd.DataFrame(rows).sort_values('VIF', ascending=False)
        df['flagged'] = df['VIF'] > threshold
        return df
    except ImportError:
        return pd.DataFrame(columns=['feature', 'VIF', 'flagged'])

def select_features(X, y, method='rfe', variance_threshold=0.01, mi_k=20, rfe_n=None, output_dir='outputs', save_selector=False, models_dir='Trained_Model'):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    var_filter = VarianceThreshold(threshold=variance_threshold)
    var_filter.fit(X)
    kept_cols = X.columns[var_filter.get_support()].tolist()
    X_var = X[kept_cols]
    vif_df = compute_vif(X_var)
    if not vif_df.empty:
        try:
            vif_df.to_csv(Path(output_dir) / 'vif_analysis.csv', index=False)
        except PermissionError:
            pass
    if method == 'variance':
        selected = kept_cols
        importance_df = pd.DataFrame({'feature': selected})
        selector = var_filter
    elif method == 'mi':
        scores = mutual_info_classif(X_var.fillna(0), y, random_state=42)
        score_series = pd.Series(scores, index=X_var.columns).sort_values(ascending=False)
        selected = score_series.head(mi_k).index.tolist()
        importance_df = pd.DataFrame({'feature': score_series.index, 'mi_score': score_series.values})
        selector = None
    elif method == 'rfe':
        n = rfe_n or max(5, X_var.shape[1] // 2)
        X_scaled = StandardScaler().fit_transform(X_var.fillna(0))
        base = LogisticRegression(C=0.1, solver='lbfgs', max_iter=500, class_weight='balanced', random_state=42)
        rfe = RFE(estimator=base, n_features_to_select=n, step=1)
        rfe.fit(X_scaled, y)
        selected = [c for c, s in zip(X_var.columns, rfe.support_) if s]
        ranking = pd.Series(rfe.ranking_, index=X_var.columns).sort_values()
        importance_df = pd.DataFrame({'feature': ranking.index, 'rfe_rank': ranking.values})
        selector = rfe
    elif method == 'all':
        scores = mutual_info_classif(X_var.fillna(0), y, random_state=42)
        score_series = pd.Series(scores, index=X_var.columns).sort_values(ascending=False)
        X_mi = X_var[score_series.head(mi_k).index.tolist()]
        n = rfe_n or max(5, X_mi.shape[1] // 2)
        X_scaled = StandardScaler().fit_transform(X_mi.fillna(0))
        base = LogisticRegression(C=0.1, solver='lbfgs', max_iter=500, class_weight='balanced', random_state=42)
        rfe = RFE(estimator=base, n_features_to_select=n, step=1)
        rfe.fit(X_scaled, y)
        selected = [c for c, s in zip(X_mi.columns, rfe.support_) if s]
        importance_df = pd.DataFrame({'feature': score_series.index, 'mi_score': score_series.values})
        selector = rfe
    else:
        raise ValueError(f"Unknown method '{method}'. Choose: variance, mi, rfe, all")
    importance_df.to_csv(Path(output_dir) / 'feature_importance.csv', index=False)
    if save_selector:
        Path(models_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(models_dir) / f'feature_selector_{method}.pkl', 'wb') as f:
            pickle.dump({'selector': selector, 'var_selector': var_filter}, f)
    return {'selected_features': selected, 'selector': selector, 'var_selector': var_filter, 'importance_df': importance_df, 'vif_df': vif_df}
if __name__ == '__main__':
    from data_preprocessing import preprocess
    from feature_engineering import build_feature_pipeline
    data = preprocess(data_dir='Datasets', apply_smote=False)
    X_eng = build_feature_pipeline().fit_transform(data['X_train'])
    res = select_features(X_eng, data['y_train'], method='rfe', rfe_n=15)
    print(f"Selected features ({len(res['selected_features'])}): {res['selected_features']}")
