import json
import pickle
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
warnings.filterwarnings('ignore')
from train_model import _IsotonicCalibratedModel
from candidate_ranking import CandidateRankingEngine, STRATEGY_PRESETS
MODELS_DIR = Path('Trained_Model')
DATA_DIR = Path('Datasets')

class CandidateInput(BaseModel):
    enrollee_id: Optional[int] = Field(None, description='Unique candidate identifier')
    city: str = Field(..., example='city_103')
    city_development_index: float = Field(..., ge=0.0, le=1.0, example=0.92)
    gender: Optional[str] = Field(None, example='Male')
    relevent_experience: str = Field(..., example='Has relevent experience')
    enrolled_university: Optional[str] = Field(None, example='no_enrollment')
    education_level: Optional[str] = Field(None, example='Graduate')
    major_discipline: Optional[str] = Field(None, example='STEM')
    experience: Optional[str] = Field(None, example='10')
    company_size: Optional[str] = Field(None, example='50-99')
    company_type: Optional[str] = Field(None, example='Pvt Ltd')
    last_new_job: Optional[str] = Field(None, example='1')
    training_hours: int = Field(..., ge=0, example=36)

    @field_validator('relevent_experience')
    @classmethod
    def validate_relevent_experience(cls, v: str) -> str:
        valid = {'Has relevent experience', 'No relevent experience'}
        if v not in valid:
            raise ValueError(f'relevent_experience must be one of {valid}')
        return v

class BatchInput(BaseModel):
    candidates: List[CandidateInput] = Field(..., min_length=1, max_length=1000)

class PredictionResult(BaseModel):
    candidate_id: Optional[int]
    progression_score: float
    prediction: int
    confidence: str
    top_shap_features: Optional[List[dict]] = None

class BatchResult(BaseModel):
    total: int
    predictions: List[PredictionResult]

class RecommendedCandidate(BaseModel):
    recommendation_rank: int
    enrollee_id: Optional[int]
    composite_score: float
    progression_score: float
    badge: str
    highlights: List[str]
    experience: Optional[str] = None
    education_level: Optional[str] = None
    training_hours: Optional[int] = None

class Top10RankingResponse(BaseModel):
    strategy: str
    total_evaluated: int
    top_10: List[RecommendedCandidate]

class CandidateDetail(BaseModel):
    rank: int
    enrollee_id: Optional[int]
    composite_score: float
    progression_score: float
    percentile: float
    intent_percent: float
    status: str
    badge: str
    experience: Optional[str] = None
    education_level: Optional[str] = None
    major_discipline: Optional[str] = None
    training_hours: Optional[int] = None
    relevent_experience: Optional[str] = None
    company_size: Optional[str] = None
    company_type: Optional[str] = None
    last_new_job: Optional[str] = None
    city_development_index: Optional[float] = None
    city: Optional[str] = None
    enrolled_university: Optional[str] = None
    gender: Optional[str] = None
    highlights: List[str] = []

class AllCandidatesResponse(BaseModel):
    total_candidates: int
    returned_count: int
    offset: int
    limit: int
    candidates: List[CandidateDetail]

class HealthResponse(BaseModel):
    status: str
    model: str
    features_count: int
_state: dict = {}
_prediction_store: dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:

        def _load_pkl(name):
            with open(MODELS_DIR / name, 'rb') as f:
                return pickle.load(f)
        _state['preprocessor'] = _load_pkl('preprocessor.pkl')
        _state['feature_pipeline'] = _load_pkl('feature_pipeline.pkl')
        _state['best_model'] = _load_pkl('best_model.pkl')
        with open(MODELS_DIR / 'selected_features.json') as f:
            _state['selected_features'] = json.load(f)
        with open(MODELS_DIR / 'model_metrics.json') as f:
            _state['best_model_name'] = json.load(f).get('best_model', 'Calibrated Classifier')
        _state['ranking_engine'] = CandidateRankingEngine()
        print('[startup] All artifacts loaded.')
    except Exception as e:
        print(f'[warning] Could not load artifacts: {e}')
    yield
    _state.clear()
app = FastAPI(title='TalentIQ Candidate Scoring and Ranking API', description='ML-powered job change prediction, SHAP attribution, and Top 10 candidate recommendation.', version='2.0.0', lifespan=lifespan)

def _load_test_csv() -> pd.DataFrame:
    path = DATA_DIR / '1_Raw_Data' / 'aug_test.csv'
    if not path.exists():
        path = DATA_DIR / 'aug_test.csv'
    if not path.exists():
        raise HTTPException(status_code=404, detail='Benchmark test dataset not found.')
    return pd.read_csv(path)

def _prepare_features(df: pd.DataFrame):
    if 'preprocessor' not in _state:
        raise HTTPException(status_code=503, detail='Model artifacts not loaded yet.')
    X = _state['preprocessor'].transform(df)
    X = _state['feature_pipeline'].transform(X)
    features = [c for c in _state['selected_features'] if c in X.columns]
    return (X[features].values, features)

def _confidence_label(prob: float) -> str:
    if prob < 0.35 or prob > 0.65:
        return 'High'
    if prob < 0.45 or prob > 0.55:
        return 'Medium'
    return 'Low'

def _get_shap_explainer(X_bg: np.ndarray):
    if _state.get('explainer') is not None:
        return _state['explainer']
    try:
        import shap
        model = _state['best_model']
        inner = getattr(model, 'base_model', model)
        model_type = type(inner).__name__.lower()
        if 'lgbm' in model_type or 'xgb' in model_type:
            explainer = shap.TreeExplainer(inner)
        else:
            explainer = shap.LinearExplainer(inner, shap.kmeans(X_bg, min(40, X_bg.shape[0])))
        _state['explainer'] = explainer
        return explainer
    except Exception:
        return None

def _compute_top_shap(X_row: np.ndarray, features: list, top_n: int=5):
    explainer = _get_shap_explainer(X_row.reshape(1, -1))
    if not explainer:
        return None
    try:
        sv = explainer.shap_values(X_row.reshape(1, -1))
        vals = sv[1][0] if isinstance(sv, list) and len(sv) > 1 else sv[0] if sv.ndim == 2 else sv
        return sorted([{'feature': f, 'value': round(float(X_row[i]), 4), 'shap': round(float(vals[i]), 6)} for i, f in enumerate(features)], key=lambda x: abs(x['shap']), reverse=True)[:top_n]
    except Exception:
        return None

def _row_to_recommendation(row: pd.Series) -> RecommendedCandidate:
    return RecommendedCandidate(recommendation_rank=int(row['recommendation_rank']), enrollee_id=int(row['enrollee_id']) if pd.notna(row.get('enrollee_id')) else None, composite_score=float(row['composite_score']), progression_score=round(float(row['progression_score']), 4), badge=str(row.get('badge', 'Qualified Applicant')), highlights=list(row.get('highlights', [])), experience=str(row.get('experience', 'N/A')), education_level=str(row.get('education_level', 'N/A')), training_hours=int(row.get('training_hours', 0)))

def _row_to_detail(row: pd.Series) -> CandidateDetail:
    return CandidateDetail(rank=int(row['rank']), enrollee_id=int(row['enrollee_id']) if pd.notna(row.get('enrollee_id')) else None, composite_score=float(row['composite_score']), progression_score=round(float(row['progression_score']), 4), percentile=float(row.get('percentile', 0.0)), intent_percent=float(row.get('intent_percent', row['progression_score'] * 100)), status=str(row.get('status', 'N/A')), badge=str(row.get('badge', 'Qualified Applicant')), experience=str(row.get('experience', 'N/A')), education_level=str(row.get('education_level', 'N/A')), major_discipline=str(row.get('major_discipline', 'N/A')), training_hours=int(row.get('training_hours', 0)), relevent_experience=str(row.get('relevent_experience', 'N/A')), company_size=str(row.get('company_size', 'N/A')), company_type=str(row.get('company_type', 'N/A')), last_new_job=str(row.get('last_new_job', 'N/A')), city_development_index=float(row.get('city_development_index', 0.0)), city=str(row.get('city', 'N/A')), enrolled_university=str(row.get('enrolled_university', 'N/A')), gender=str(row.get('gender', 'N/A')), highlights=list(row.get('highlights', [])))

def _score_df(df: pd.DataFrame, strategy: str='balanced') -> pd.DataFrame:
    X, _ = _prepare_features(df)
    df['progression_score'] = _state['best_model'].predict_proba(X)[:, 1]
    engine = _state.get('ranking_engine', CandidateRankingEngine())
    return engine.score_candidates(df, strategy=strategy)

@app.get('/', tags=['General'])
def root():
    return {'service': 'TalentIQ Candidate Scoring and Ranking API', 'version': '2.0.0', 'documentation': '/docs'}

@app.get('/health', response_model=HealthResponse, tags=['Diagnostics'])
def health():
    if 'best_model' not in _state:
        raise HTTPException(status_code=503, detail='Service unavailable.')
    return HealthResponse(status='operational', model=_state.get('best_model_name', 'unknown'), features_count=len(_state.get('selected_features', [])))

@app.post('/predict', response_model=PredictionResult, tags=['Inference'])
def predict_candidate(candidate: CandidateInput):
    df = pd.DataFrame([candidate.model_dump()])
    X, feats = _prepare_features(df)
    prob = float(_state['best_model'].predict_proba(X)[0, 1])
    top_shap = _compute_top_shap(X[0], feats)
    result = PredictionResult(candidate_id=candidate.enrollee_id, progression_score=round(prob, 4), prediction=int(prob >= 0.5), confidence=_confidence_label(prob), top_shap_features=top_shap)
    if candidate.enrollee_id is not None:
        _prediction_store[candidate.enrollee_id] = {'result': result.model_dump(), 'X_row': X[0].tolist(), 'features': feats}
    return result

@app.post('/predict/batch', response_model=BatchResult, tags=['Inference'])
def predict_batch_candidates(batch: BatchInput):
    df = pd.DataFrame([c.model_dump() for c in batch.candidates])
    X, _ = _prepare_features(df)
    probs = _state['best_model'].predict_proba(X)[:, 1]
    results = [PredictionResult(candidate_id=c.enrollee_id, progression_score=round(float(probs[i]), 4), prediction=int(probs[i] >= 0.5), confidence=_confidence_label(float(probs[i]))) for i, c in enumerate(batch.candidates)]
    return BatchResult(total=len(results), predictions=results)

@app.post('/rank/top10', response_model=Top10RankingResponse, tags=['Talent Ranking'])
def rank_top_10(batch: BatchInput, strategy: str=Query('balanced', regex='^(balanced|senior|fast_hire|high_growth)$')):
    df = pd.DataFrame([c.model_dump() for c in batch.candidates])
    ranked = _score_df(df, strategy=strategy)
    top10 = ranked.head(10).copy()
    top10['recommendation_rank'] = range(1, len(top10) + 1)
    return Top10RankingResponse(strategy=strategy, total_evaluated=len(df), top_10=[_row_to_recommendation(row) for _, row in top10.iterrows()])

@app.get('/candidates/top10', response_model=Top10RankingResponse, tags=['Talent Ranking'])
def benchmark_top_10(strategy: str=Query('balanced', regex='^(balanced|senior|fast_hire|high_growth)$')):
    df = _load_test_csv()
    ranked = _score_df(df, strategy=strategy)
    top10 = ranked.head(10).copy()
    top10['recommendation_rank'] = range(1, len(top10) + 1)
    return Top10RankingResponse(strategy=strategy, total_evaluated=len(df), top_10=[_row_to_recommendation(row) for _, row in top10.iterrows()])

@app.get('/candidates/all', response_model=AllCandidatesResponse, tags=['Talent Ranking'])
def get_all_candidates(limit: int=Query(100, ge=1, le=25000), offset: int=Query(0, ge=0), strategy: str=Query('balanced', regex='^(balanced|senior|fast_hire|high_growth)$')):
    df = _load_test_csv()
    ranked = _score_df(df, strategy=strategy)
    sliced = ranked.iloc[offset:offset + limit]
    return AllCandidatesResponse(total_candidates=len(ranked), returned_count=len(sliced), offset=offset, limit=limit, candidates=[_row_to_detail(row) for _, row in sliced.iterrows()])

@app.get('/candidates/{candidate_id}', response_model=CandidateDetail, tags=['Talent Ranking'])
def get_candidate_by_id(candidate_id: int):
    df = _load_test_csv()
    if df[df['enrollee_id'] == candidate_id].empty:
        raise HTTPException(status_code=404, detail=f'Candidate #{candidate_id} not found.')
    ranked = _score_df(df)
    cand_row = ranked[ranked['enrollee_id'] == candidate_id].iloc[0]
    return _row_to_detail(cand_row)

@app.get('/explain/{candidate_id}', tags=['Explainability'])
def explain_candidate(candidate_id: int):
    if candidate_id not in _prediction_store:
        raise HTTPException(status_code=404, detail=f'No cached prediction for ID={candidate_id}. Call /predict first.')
    cached = _prediction_store[candidate_id]
    X_row = np.array(cached['X_row'])
    top_shap = _compute_top_shap(X_row, cached['features'])
    return JSONResponse(content={'candidate_id': candidate_id, 'prediction': cached['result']['prediction'], 'progression_score': cached['result']['progression_score'], 'confidence': cached['result']['confidence'], 'shap_explanation': top_shap})
if __name__ == '__main__':
    import uvicorn
    uvicorn.run('app:app', host='0.0.0.0', port=8000, reload=True)
