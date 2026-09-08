import os
import sys
import json
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))
try:
    import lightgbm
    import sklearn
    import pandas as pd
    import numpy as np
except ModuleNotFoundError as e:
    missing_pkg = e.name
    print('=' * 60)
    print(f"[ENVIRONMENT NOTICE] Missing dependency: '{missing_pkg}'")
    print('=' * 60)
    print("You are running with a Python environment that doesn't have the project packages.")
    print('\nHow to fix:')
    print('1. If testing on this machine, use the project virtual environment:')
    print('   ..\\venv\\Scripts\\python test_model_loading.py')
    print('   or simply double-click: run_test.bat')
    print('\n2. If testing in a new environment, install the requirements:')
    print('   pip install -r requirements.txt')
    print('=' * 60)
    sys.exit(1)
from train_model import _IsotonicCalibratedModel

def test_loading():
    print('=' * 60)
    print('Testing Deliverable 3: Trained Model (.pkl)')
    print('=' * 60)
    model_path = current_dir / 'best_model.pkl'
    preprocessor_path = current_dir / 'preprocessor.pkl'
    pipeline_path = current_dir / 'feature_pipeline.pkl'
    features_path = current_dir / 'selected_features.json'
    metrics_path = current_dir / 'model_metrics.json'
    for p in [model_path, preprocessor_path, pipeline_path, features_path, metrics_path]:
        if not p.exists():
            print(f'[ERROR] Missing file: {p.name}')
            return False
        print(f'[OK] Found {p.name} ({p.stat().st_size:,} bytes)')
    print('\nLoading model artifacts from disk...')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(preprocessor_path, 'rb') as f:
        preprocessor = pickle.load(f)
    with open(pipeline_path, 'rb') as f:
        pipeline = pickle.load(f)
    with open(features_path, 'r') as f:
        selected_features = json.load(f)
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    print(f"[SUCCESS] Champion model loaded: {metrics.get('best_model', 'Calibrated Model')}")
    print(f'[SUCCESS] Total selected features: {len(selected_features)}')
    import random
    test_csv_path = parent_dir / 'Datasets' / '1_Raw_Data' / 'aug_test.csv'
    if not test_csv_path.exists():
        test_csv_path = parent_dir / 'Datasets' / 'aug_test.csv'
    if test_csv_path.exists():
        df_all = pd.read_csv(test_csv_path)
        sample_candidate = df_all.sample(n=1).copy()
        cand_source = f'Randomly drawn from test dataset ({len(df_all):,} candidates)'
    else:
        sample_candidate = pd.DataFrame([{'enrollee_id': random.randint(10000, 99999), 'city': random.choice(['city_103', 'city_21', 'city_16', 'city_114', 'city_160', 'city_136', 'city_61']), 'city_development_index': round(random.uniform(0.5, 0.95), 3), 'gender': random.choice(['Male', 'Female', 'Other']), 'relevent_experience': random.choice(['Has relevent experience', 'No relevent experience']), 'enrolled_university': random.choice(['no_enrollment', 'Full time course', 'Part time course']), 'education_level': random.choice(['Graduate', 'Masters', 'Phd', 'High School']), 'major_discipline': random.choice(['STEM', 'Humanities', 'Business Degree', 'Other']), 'experience': random.choice(['<1', '1', '3', '5', '8', '10', '15', '20', '>20']), 'company_size': random.choice(['<10', '10/49', '50-99', '100-500', '500-999', '1000-4999', '10000+']), 'company_type': random.choice(['Pvt Ltd', 'Funded Startup', 'Public Sector', 'Early Stage Startup']), 'last_new_job': random.choice(['never', '1', '2', '3', '4', '>4']), 'training_hours': random.randint(10, 260)}])
        cand_source = 'Synthetically generated random applicant profile'
    cand = sample_candidate.iloc[0]
    cid = cand.get('enrollee_id', 'N/A')
    print('\n' + '-' * 60)
    print(f'SIMULATING RANDOM CANDIDATE #{cid}')
    print(f'Source: {cand_source}')
    print('-' * 60)
    print(f"  • Education Level     : {cand.get('education_level', 'N/A')}")
    print(f"  • Major Discipline    : {cand.get('major_discipline', 'N/A')}")
    print(f"  • Experience (Years)  : {cand.get('experience', 'N/A')} yrs")
    print(f"  • Relevant Experience : {cand.get('relevent_experience', 'N/A')}")
    print(f"  • Training Hours      : {cand.get('training_hours', 0)} hrs")
    print(f"  • City Location       : {cand.get('city', 'N/A')} (CDI: {cand.get('city_development_index', 0.0):.3f})")
    print(f"  • Company Size / Type : {cand.get('company_size', 'N/A')} | {cand.get('company_type', 'N/A')}")
    print(f"  • Last New Job        : {cand.get('last_new_job', 'N/A')}")
    X_prep = preprocessor.transform(sample_candidate)
    X_feat = pipeline.transform(X_prep)
    X_sel = X_feat[[c for c in selected_features if c in X_feat.columns]].values
    probability = float(model.predict_proba(X_sel)[0, 1])
    prediction = int(probability >= 0.5)
    if probability < 0.35 or probability > 0.65:
        confidence = 'High'
    elif probability < 0.45 or probability > 0.55:
        confidence = 'Medium'
    else:
        confidence = 'Low'
    print('\n' + '=' * 60)
    print('MODEL INFERENCE RESULTS')
    print('=' * 60)
    print(f'  Candidate ID        : #{cid}')
    print(f'  Transition Intent   : {probability * 100:.2f}% (Calibrated Score: {probability:.4f})')
    print(f"  Predicted Decision  : {prediction} -> {('[LIKELY TO SEEK JOB CHANGE]' if prediction == 1 else '[STABLE IN CURRENT ROLE]')}")
    print(f'  Prediction Confidence: {confidence}')
    print('=' * 60)
    print('[SUCCESS] Deliverable 3 model (.pkl) verified with random candidate!')
    return True
if __name__ == '__main__':
    test_loading()
