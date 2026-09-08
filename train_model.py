import json
import pickle
import warnings
from pathlib import Path
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import optuna
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

class _IsotonicCalibratedModel:

    def __init__(self, base_model):
        self.base_model = base_model

    def fit(self, X_val, y_val):
        raw_probs = self.base_model.predict_proba(X_val)[:, 1]
        self._calibrator = IsotonicRegression(out_of_bounds='clip')
        self._calibrator.fit(raw_probs, y_val)
        return self

    def predict_proba(self, X):
        raw = self.base_model.predict_proba(X)[:, 1]
        cal = self._calibrator.predict(raw)
        return np.column_stack([1.0 - cal, cal])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

def _lgbm_objective(trial, X_tr, y_tr, X_val, y_val):
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'random_state': 42,
        'n_estimators': trial.suggest_int('n_estimators', 100, 800),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-08, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-08, 10.0, log=True),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 5.0)
    }
    model = lgb.LGBMClassifier(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
    y_pred_prob = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, y_pred_prob)

def _xgb_objective(trial, X_tr, y_tr, X_val, y_val):
    num_zeros = (y_tr == 0).sum()
    num_ones = max((y_tr == 1).sum(), 1)
    scale = float(num_zeros / num_ones)
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'verbosity': 0,
        'random_state': 42,
        'use_label_encoder': False,
        'n_estimators': trial.suggest_int('n_estimators', 100, 800),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-08, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-08, 10.0, log=True),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, scale + 1)
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    y_pred_prob = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, y_pred_prob)

def _lr_objective(trial, X_tr, y_tr, X_val, y_val):
    params = {
        'C': trial.suggest_float('C', 0.0001, 100.0, log=True),
        'solver': trial.suggest_categorical('solver', ['lbfgs', 'saga']),
        'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
        'max_iter': 1000,
        'random_state': 42
    }
    model = LogisticRegression(**params)
    model.fit(X_tr, y_tr)
    y_pred_prob = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, y_pred_prob)

def _build_lgbm(params):
    return lgb.LGBMClassifier(**params, random_state=42, verbosity=-1)

def _build_xgb(params):
    return xgb.XGBClassifier(**params, random_state=42, verbosity=0, use_label_encoder=False)

def _build_lr(params):
    return LogisticRegression(**params, max_iter=1000, random_state=42)

def _evaluate(model, X, y, name):
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= 0.5).astype(int)
    return {'model': name, 'auc_roc': round(float(roc_auc_score(y, prob)), 4), 'f1': round(float(f1_score(y, pred)), 4), 'precision': round(float(precision_score(y, pred, zero_division=0)), 4), 'recall': round(float(recall_score(y, pred)), 4)}

def train(data_dir='Datasets', n_trials=50, feature_selection_method='rfe', rfe_n=15, save_dir='Trained_Model', output_dir='outputs'):
    from data_preprocessing import preprocess
    from feature_engineering import build_feature_pipeline
    from feature_selection import select_features
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    prep = preprocess(data_dir=data_dir, apply_smote=True, save_preprocessor=True, models_dir=save_dir)
    X_tr, y_tr = (prep['X_train'], prep['y_train'])
    X_val, y_val = (prep['X_val'], prep['y_val'])
    feat_pipe = build_feature_pipeline()
    X_tr = feat_pipe.fit_transform(X_tr)
    X_val = feat_pipe.transform(X_val)
    with open(Path(save_dir) / 'feature_pipeline.pkl', 'wb') as f:
        pickle.dump(feat_pipe, f)
    sel = select_features(X_tr, y_tr, method=feature_selection_method, rfe_n=rfe_n, output_dir=output_dir, save_selector=True, models_dir=save_dir)
    selected = sel['selected_features']
    with open(Path(save_dir) / 'selected_features.json', 'w') as f:
        json.dump(selected, f)
    X_tr_s = X_tr[selected].values
    X_val_s = X_val[[c for c in selected if c in X_val.columns]].values
    y_tr_arr, y_val_arr = (y_tr.values, y_val.values)
    model_configs = [('LightGBM', _lgbm_objective, _build_lgbm), ('XGBoost', _xgb_objective, _build_xgb), ('LogisticRegression', _lr_objective, _build_lr)]
    trained_models, all_metrics = ({}, [])
    for name, obj_fn, builder_fn in model_configs:
        print(f'Tuning {name}...')
        study = optuna.create_study(direction='maximize', study_name=name)
        study.optimize(lambda trial, o=obj_fn: o(trial, X_tr_s, y_tr_arr, X_val_s, y_val_arr), n_trials=n_trials, show_progress_bar=False)
        raw_model = builder_fn(study.best_params)
        raw_model.fit(X_tr_s, y_tr_arr)
        cal_model = _IsotonicCalibratedModel(raw_model)
        cal_model.fit(X_val_s, y_val_arr)
        metrics = _evaluate(cal_model, X_val_s, y_val_arr, name)
        all_metrics.append(metrics)
        trained_models[name] = cal_model
        print(f"  AUC-ROC: {metrics['auc_roc']}")
        slug = name.lower().replace(' ', '_')
        with open(Path(save_dir) / f'{slug}_model.pkl', 'wb') as f:
            pickle.dump(cal_model, f)
    best_name = max(all_metrics, key=lambda m: m['auc_roc'])['model']
    print(f'\nBest model: {best_name}')
    with open(Path(save_dir) / 'best_model.pkl', 'wb') as f:
        pickle.dump(trained_models[best_name], f)
    metadata = {'best_model': best_name, 'selected_features': selected, 'metrics': all_metrics}
    with open(Path(save_dir) / 'model_metrics.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    return {'models': trained_models, 'metrics': all_metrics, 'best': best_name, 'features': selected}
if __name__ == '__main__':
    res = train(n_trials=10)
    print(f"Training completed. Champion model: {res['best']}")
