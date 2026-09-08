import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

class FeatureEngineer:

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        exp = X['experience'].astype(float)
        edu = X['education_level'].astype(float)
        hrs = X['training_hours'].astype(float)
        comp = X['company_size'].astype(float)
        lnj = X['last_new_job'].astype(float)
        cdi = X['city_development_index'].astype(float)
        X['training_intensity'] = hrs / (exp + 1.0)
        X['edu_vs_exp_ratio'] = edu / (exp + 1.0)
        X['career_stability'] = comp / (lnj + 1.0)
        X['seniority_score'] = exp * cdi
        X['edu_exp_interaction'] = edu * exp
        if 'major_discipline_STEM' in X.columns:
            X['stem_flag'] = X['major_discipline_STEM'].astype(int)
        elif 'major_discipline' in X.columns:
            X['stem_flag'] = (X['major_discipline'] == 'STEM').astype(int)
        else:
            X['stem_flag'] = 0
        return X

class FeaturePipeline:

    def __init__(self, steps):
        self.steps = steps

    def fit(self, X, y=None):
        for _, t in self.steps:
            t.fit(X, y)
            X = t.transform(X)
        return self

    def transform(self, X):
        for _, t in self.steps:
            X = t.transform(X)
        return X

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def get_params(self, deep=True):
        return {'steps': self.steps}

def build_feature_pipeline():
    return FeaturePipeline(steps=[('engineer', FeatureEngineer())])
ENGINEERED_FEATURE_NAMES = ['training_intensity', 'edu_vs_exp_ratio', 'career_stability', 'seniority_score', 'edu_exp_interaction', 'stem_flag']
if __name__ == '__main__':
    from data_preprocessing import preprocess
    data = preprocess(data_dir='Datasets', apply_smote=False)
    pipe = build_feature_pipeline()
    transformed = pipe.fit_transform(data['X_train'])
    print(f'Engineered dataset shape: {transformed.shape}')
