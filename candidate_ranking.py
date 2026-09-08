import json
import pickle
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

# --- Step 1: Strategy Presets and Maps ---
STRATEGY_PRESETS = {
    'balanced': {'intent': 0.35, 'experience': 0.25, 'education': 0.20, 'training': 0.10, 'relevance': 0.10},
    'senior': {'intent': 0.20, 'experience': 0.40, 'education': 0.25, 'training': 0.05, 'relevance': 0.10},
    'fast_hire': {'intent': 0.60, 'experience': 0.15, 'education': 0.10, 'training': 0.05, 'relevance': 0.10},
    'high_growth': {'intent': 0.25, 'experience': 0.15, 'education': 0.15, 'training': 0.35, 'relevance': 0.10}
}
EDU_SCORE_MAP = {
    'Phd': 1.0,
    'Masters': 0.88,
    'Graduate': 0.70,
    'High School': 0.40,
    'Primary School': 0.20
}


# --- Step 2: Experience String Parser ---
def parse_experience(val):
    if pd.isna(val):
        return 5.0
    s = str(val).strip()
    if s == '<1':
        return 0.5
    if s == '>20':
        return 22.0
    try:
        return float(s)
    except ValueError:
        return 5.0


# --- Step 3: Candidate Ranking Engine ---
class CandidateRankingEngine:

    def __init__(self, default_strategy='balanced'):
        self.default_strategy = default_strategy

    def score_candidates(self, df, strategy='balanced', custom_weights=None):
        if 'progression_score' not in df.columns:
            raise ValueError("DataFrame must have a 'progression_score' column.")
        if custom_weights is not None:
            weights = custom_weights
        else:
            weights = STRATEGY_PRESETS.get(strategy, STRATEGY_PRESETS['balanced'])

        out = df.copy()
        out['score_intent'] = out['progression_score'].clip(0.0, 1.0)

        if 'experience' in out.columns:
            exp_col = out['experience'].apply(parse_experience)
        else:
            exp_col = pd.Series(5.0, index=out.index)
        out['exp_years'] = exp_col
        out['score_experience'] = (exp_col / 20.0).clip(0.0, 1.0)

        if 'education_level' in out.columns:
            edu_base = out['education_level'].map(EDU_SCORE_MAP).fillna(0.6)
        else:
            edu_base = pd.Series(0.6, index=out.index)

        stem_bonus = 0.0
        if 'major_discipline' in out.columns:
            is_stem = out['major_discipline'].str.upper() == 'STEM'
            stem_bonus = is_stem.astype(float) * 0.1
        out['score_education'] = (edu_base + stem_bonus).clip(0.0, 1.0)

        if 'training_hours' in out.columns:
            hrs = out['training_hours'].astype(float)
            out['score_training'] = (hrs / 200.0).clip(0.0, 1.0)
        else:
            out['score_training'] = 0.5

        if 'relevent_experience' in out.columns:
            def check_rel(v):
                if str(v).lower().startswith('has'):
                    return 1.0
                return 0.35
            out['score_relevance'] = out['relevent_experience'].apply(check_rel)
        else:
            out['score_relevance'] = 0.5

        w_intent = weights['intent']
        w_exp = weights['experience']
        w_edu = weights['education']
        w_train = weights['training']
        w_rel = weights['relevance']

        composite = (
            w_intent * out['score_intent'] +
            w_exp * out['score_experience'] +
            w_edu * out['score_education'] +
            w_train * out['score_training'] +
            w_rel * out['score_relevance']
        ) * 100.0
        out['composite_score'] = composite.round(2)
        out['rank'] = out['composite_score'].rank(ascending=False, method='min').astype(int)
        out['percentile'] = (out['composite_score'].rank(pct=True) * 100.0).round(1)
        out['intent_percent'] = (out['score_intent'] * 100.0).round(1)
        out['experience_percent'] = (out['score_experience'] * 100.0).round(1)
        out['education_percent'] = (out['score_education'] * 100.0).round(1)
        out['training_percent'] = (out['score_training'] * 100.0).round(1)
        out['relevance_percent'] = (out['score_relevance'] * 100.0).round(1)
        out['badge'] = out.apply(self._badge, axis=1)
        out['highlights'] = out.apply(self._highlights, axis=1)
        return out.sort_values('composite_score', ascending=False).reset_index(drop=True)

    def recommend_top_k(self, df, k=10, strategy='balanced', custom_weights=None):
        ranked = self.score_candidates(df, strategy=strategy, custom_weights=custom_weights)
        top = ranked.head(k).copy()
        top['recommendation_rank'] = range(1, len(top) + 1)
        return top

    @staticmethod
    def _badge(row):
        score = row.get('composite_score', 0)
        exp = row.get('exp_years', 0)
        intent = row.get('score_intent', 0)
        hrs = row.get('training_hours', 0)
        if score >= 85 and exp >= 10:
            return 'Executive Talent'
        if exp >= 12:
            return 'Senior Specialist'
        if hrs >= 100 and intent >= 0.7:
            return 'High-Velocity Learner'
        if intent >= 0.8:
            return 'Immediate Mover'
        if score >= 75:
            return 'Strong Match'
        return 'Qualified Applicant'

    @staticmethod
    def _highlights(row):
        items = []
        intent = row.get('score_intent', 0)
        exp = row.get('exp_years', 0)
        edu = row.get('education_level', 'N/A')
        major = row.get('major_discipline', 'N/A')
        hrs = row.get('training_hours', 0)
        rel = row.get('relevent_experience', '')
        if intent >= 0.75:
            items.append(f'High transition readiness ({intent * 100:.0f}%)')
        elif intent >= 0.5:
            items.append('Open to progression')
        if exp >= 15:
            items.append(f'Deep seniority ({int(exp)}+ yrs)')
        elif exp >= 8:
            items.append(f'Mid-to-Senior ({int(exp)} yrs)')
        elif exp > 0:
            items.append(f'{int(exp)} yrs exp')
        if edu in ('Masters', 'Phd'):
            items.append(f'Advanced degree ({edu})')
        if major == 'STEM':
            items.append('STEM discipline')
        if hrs >= 80:
            items.append(f'Committed upskiller ({int(hrs)} hrs)')
        if 'has' in str(rel).lower():
            items.append('Domain-relevant experience')
        return items[:4]


# --- Step 4: Standalone Ranking Test ---
if __name__ == '__main__':
    print('Testing CandidateRankingEngine...')
    models_dir = Path('Trained_Model')
    data_dir = Path('Datasets')
    test_csv = data_dir / '1_Raw_Data' / 'aug_test.csv'
    if not test_csv.exists():
        test_csv = data_dir / 'aug_test.csv'
    if (models_dir / 'best_model.pkl').exists() and test_csv.exists():
        from train_model import _IsotonicCalibratedModel
        with open(models_dir / 'preprocessor.pkl', 'rb') as f:
            prep = pickle.load(f)
        with open(models_dir / 'feature_pipeline.pkl', 'rb') as f:
            pipe = pickle.load(f)
        with open(models_dir / 'best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open(models_dir / 'selected_features.json') as f:
            features = json.load(f)
        raw = pd.read_csv(test_csv).head(100)
        X = prep.transform(raw)
        X = pipe.transform(X)
        raw['progression_score'] = model.predict_proba(X[features].values)[:, 1]
        engine = CandidateRankingEngine()
        top10 = engine.recommend_top_k(raw, k=10)
        print(f'\nTop 10 Applicants (out of {len(raw)}):')
        cols = ['recommendation_rank', 'enrollee_id', 'composite_score', 'progression_score', 'badge', 'highlights']
        print(top10[cols].to_string(index=False))
    else:
        print('Model or dataset not found. Skipping live demonstration.')
