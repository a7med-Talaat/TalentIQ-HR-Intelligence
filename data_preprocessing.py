import pickle
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
warnings.filterwarnings('ignore')
TARGET_COL = 'target'
DROP_COLS = ['enrollee_id']
CITY_COL = 'city'
NUMERIC_COLS = ['city_development_index', 'training_hours']
NOMINAL_COLS = ['gender', 'enrolled_university', 'major_discipline', 'company_type']
ORDINAL_COLS = ['education_level', 'experience', 'company_size', 'last_new_job']
EDUCATION_MAP = {
    'Primary School': 0,
    'High School': 1,
    'Graduate': 2,
    'Masters': 3,
    'Phd': 4
}
EXPERIENCE_MAP = {'<1': 0}
for i in range(1, 21):
    EXPERIENCE_MAP[str(i)] = i
EXPERIENCE_MAP['>20'] = 21

COMPANY_SIZE_MAP = {
    '<10': 0,
    '10/49': 1,
    '50-99': 2,
    '100-500': 3,
    '500-999': 4,
    '1000-4999': 5,
    '5000-9999': 6,
    '10000+': 7
}
LAST_NEW_JOB_MAP = {
    'never': 0,
    '1': 1,
    '2': 2,
    '3': 3,
    '4': 4,
    '>4': 5
}
ORDINAL_MAPS = [EDUCATION_MAP, EXPERIENCE_MAP, COMPANY_SIZE_MAP, LAST_NEW_JOB_MAP]
BINARY_MAP = {'Has relevent experience': 1, 'No relevent experience': 0}

class PreprocessingPipeline:

    def fit(self, X, y):
        X = X.drop(columns=DROP_COLS, errors='ignore').copy()
        self._ordinal_medians = {}
        for col, mapping in zip(ORDINAL_COLS, ORDINAL_MAPS):
            mapped = X[col].map(mapping)
            self._ordinal_medians[col] = float(np.nanmedian(mapped.dropna()))
        self._numeric_medians = {col: float(X[col].median()) for col in NUMERIC_COLS}
        smoothing = 10.0
        global_mean = float(y.mean())
        self._city_global_mean = global_mean
        stats = pd.DataFrame({'city': X[CITY_COL], '_y': y}).groupby('city')['_y']
        self._city_encoding = ((stats.sum() + smoothing * global_mean) / (stats.count() + smoothing)).to_dict()
        self._ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        self._ohe.fit(X[NOMINAL_COLS].fillna('Unknown'))
        self._ohe_cols = self._ohe.get_feature_names_out(NOMINAL_COLS).tolist()
        return self

    def transform(self, X):
        X = X.drop(columns=DROP_COLS, errors='ignore').copy()
        for col, mapping in zip(ORDINAL_COLS, ORDINAL_MAPS):
            X[col] = X[col].map(mapping).fillna(self._ordinal_medians[col])
        X['relevent_experience'] = X['relevent_experience'].map(BINARY_MAP).fillna(0).astype(int)
        for col in NOMINAL_COLS:
            X[col] = X[col].fillna('Unknown')
        for col, median in self._numeric_medians.items():
            X[col] = X[col].fillna(median)
        X[CITY_COL] = X[CITY_COL].map(self._city_encoding).fillna(self._city_global_mean)
        ohe_arr = self._ohe.transform(X[NOMINAL_COLS])
        ohe_df = pd.DataFrame(ohe_arr, columns=self._ohe_cols, index=X.index)
        X = pd.concat([X.drop(columns=NOMINAL_COLS), ohe_df], axis=1)
        self._last_col_names = X.columns.tolist()
        return X

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        result = self.transform(X)
        self._last_col_names = result.columns.tolist()
        return result

    def get_feature_names_out(self):
        return getattr(self, '_last_col_names', [])

def load_raw(data_dir='Datasets'):
    root = Path(data_dir)
    train_path = root / '1_Raw_Data' / 'aug_train.csv'
    test_path = root / '1_Raw_Data' / 'aug_test.csv'
    if not train_path.exists():
        train_path = root / 'aug_train.csv'
    if not test_path.exists():
        test_path = root / 'aug_test.csv'
    return (pd.read_csv(train_path), pd.read_csv(test_path))

def preprocess(data_dir='Datasets', test_size=0.2, random_state=42, apply_smote=True, smote_k=5, save_preprocessor=False, models_dir='Trained_Model'):
    train_raw, test_raw = load_raw(data_dir)
    X = train_raw.drop(columns=[TARGET_COL])
    y = train_raw[TARGET_COL].astype(int)
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    preprocessor = PreprocessingPipeline()
    X_tr = preprocessor.fit_transform(X_tr, y_tr)
    X_val = preprocessor.transform(X_val)
    X_test = preprocessor.transform(test_raw)
    feature_names = X_tr.columns.tolist()
    if apply_smote:
        from imblearn.over_sampling import SMOTE
        X_arr, y_arr = SMOTE(k_neighbors=smote_k, random_state=random_state).fit_resample(X_tr.values, y_tr.values)
        X_tr = pd.DataFrame(X_arr, columns=feature_names)
        y_tr = pd.Series(y_arr, name=TARGET_COL)
    if save_preprocessor:
        out_dir = Path(models_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / 'preprocessor.pkl', 'wb') as f:
            pickle.dump(preprocessor, f)
    return {'X_train': X_tr, 'X_val': X_val, 'y_train': y_tr, 'y_val': y_val, 'X_test': X_test, 'preprocessor': preprocessor, 'feature_names': feature_names}
if __name__ == '__main__':
    res = preprocess(save_preprocessor=False)
    print(f"Preprocessed train shape: {res['X_train'].shape}, Features: {len(res['feature_names'])}")
