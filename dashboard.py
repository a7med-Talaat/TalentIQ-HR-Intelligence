import json
import pickle
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
warnings.filterwarnings('ignore')
from train_model import _IsotonicCalibratedModel
from candidate_ranking import CandidateRankingEngine, STRATEGY_PRESETS
MODELS_DIR = Path('Trained_Model')
OUTPUTS_DIR = Path('outputs')
DATA_DIR = Path('Datasets')
st.set_page_config(page_title='TalentIQ · Candidate Intelligence', page_icon='⚡', layout='wide', initial_sidebar_state='expanded')
st.markdown('\n<style>\n    @import url(\'https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap\');\n    html, body, [class*="css"] { font-family: \'Plus Jakarta Sans\', sans-serif; }\n    .stApp { background: radial-gradient(circle at 10% 20%, #0d1322 0%, #080c15 90%); color: #f1f5f9; }\n\n    /* KPI Summary Cards */\n    .kpi-container { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1rem; margin-bottom: 1.5rem; }\n    .kpi-card { background: #131b2e; border: 1px solid #232f48; border-radius: 14px; padding: 1.25rem 1.5rem; }\n    .kpi-label { font-size: 0.8rem; text-transform: uppercase; color: #94a3b8; font-weight: 600; }\n    .kpi-value { font-size: 2rem; font-weight: 800; color: #ffffff; margin: 0.2rem 0; }\n    .kpi-sub { font-size: 0.8rem; color: #34d399; font-weight: 600; }\n\n    /* Top 3 Podium Cards */\n    .podium-card { border-radius: 16px; padding: 1.25rem; margin-bottom: 1rem; background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.1); }\n    .podium-1 { border: 1.5px solid #f59e0b; background: linear-gradient(180deg, rgba(245, 158, 11, 0.08) 0%, rgba(255, 255, 255, 0.02) 100%); }\n    .podium-2 { border: 1.5px solid #94a3b8; background: linear-gradient(180deg, rgba(148, 163, 184, 0.08) 0%, rgba(255, 255, 255, 0.02) 100%); }\n    .podium-3 { border: 1.5px solid #d97706; background: linear-gradient(180deg, rgba(217, 119, 6, 0.08) 0%, rgba(255, 255, 255, 0.02) 100%); }\n\n    /* Badges & Tags */\n    .pill-tag { display: inline-block; background: rgba(99, 102, 241, 0.12); color: #c7d2fe; border: 1px solid rgba(99, 102, 241, 0.25); border-radius: 6px; padding: 0.15rem 0.5rem; font-size: 0.72rem; margin: 0.15rem 0.2rem 0.15rem 0; }\n    .badge-gold   { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); padding: 0.2rem 0.6rem; border-radius: 999px; font-weight: 700; font-size: 0.75rem; }\n    .badge-silver { background: rgba(148, 163, 184, 0.2); color: #e2e8f0; border: 1px solid rgba(148, 163, 184, 0.4); padding: 0.2rem 0.6rem; border-radius: 999px; font-weight: 700; font-size: 0.75rem; }\n    .badge-bronze { background: rgba(217, 119, 6, 0.2); color: #f97316; border: 1px solid rgba(217, 119, 6, 0.4); padding: 0.2rem 0.6rem; border-radius: 999px; font-weight: 700; font-size: 0.75rem; }\n</style>\n', unsafe_allow_html=True)

@st.cache_resource(show_spinner='Loading trained model...')
def load_artifacts():
    try:

        def _load_pkl(name):
            with open(MODELS_DIR / name, 'rb') as f:
                return pickle.load(f)
        with open(MODELS_DIR / 'selected_features.json') as f:
            selected_features = json.load(f)
        with open(MODELS_DIR / 'model_metrics.json') as f:
            metadata = json.load(f)
        return {'preprocessor': _load_pkl('preprocessor.pkl'), 'feature_pipeline': _load_pkl('feature_pipeline.pkl'), 'best_model': _load_pkl('best_model.pkl'), 'selected_features': selected_features, 'best_model_name': metadata.get('best_model', 'Calibrated Classifier'), 'metrics': metadata.get('metrics', [])}
    except Exception:
        return None

@st.cache_resource(show_spinner='Initializing SHAP explainer...')
def load_shap_explainer(_model, _X_background: np.ndarray):
    try:
        import shap
        inner = getattr(_model, 'base_model', _model)
        mtype = type(inner).__name__.lower()
        if 'lgbm' in mtype or 'xgb' in mtype:
            return shap.TreeExplainer(inner)
        sample = shap.kmeans(_X_background, min(40, _X_background.shape[0]))
        return shap.LinearExplainer(inner, sample)
    except Exception:
        return None

@st.cache_data(show_spinner='Running candidate scoring...')
def run_inference(df: pd.DataFrame, _artifacts: dict) -> pd.DataFrame:
    X_prep = _artifacts['preprocessor'].transform(df)
    X_feat = _artifacts['feature_pipeline'].transform(X_prep)
    feats = _artifacts['selected_features']
    X_mat = X_feat[[c for c in feats if c in X_feat.columns]].values
    probs = _artifacts['best_model'].predict_proba(X_mat)[:, 1]
    res = df.copy()
    res['progression_score'] = np.round(probs, 4)
    res['prediction'] = (probs >= 0.5).astype(int)
    res['status'] = np.where(probs >= 0.5, 'Ready for Transition', 'Stable in Role')
    res['_X'] = list(X_mat)
    res['_feats'] = [feats] * len(df)
    return res

def clean_val(val, default='Not Specified') -> str:
    if val is None or pd.isna(val):
        return default
    s = str(val).strip()
    if s.lower() in ('nan', 'none', '', 'null'):
        return default
    if '_' in s and (not s.lower().startswith('city_')):
        return s.replace('_', ' ').title()
    return s

def render_kpi_bar(total: int, top10_avg: float, ready_count: int, senior_count: int):
    pct_ready = ready_count / max(total, 1) * 100
    st.markdown(f'\n    <div class="kpi-container">\n        <div class="kpi-card">\n            <div class="kpi-label">Active Candidate Pool</div>\n            <div class="kpi-value">{total:,}</div>\n            <div class="kpi-sub">Applicants Evaluated</div>\n        </div>\n        <div class="kpi-card">\n            <div class="kpi-label">Top 10 Benchmark Match</div>\n            <div class="kpi-value">{top10_avg:.1f}<span style="font-size: 1.1rem; color: #94a3b8;">%</span></div>\n            <div class="kpi-sub">Average Composite Fit</div>\n        </div>\n        <div class="kpi-card">\n            <div class="kpi-label">High Transition Intent</div>\n            <div class="kpi-value">{ready_count:,}</div>\n            <div class="kpi-sub">{pct_ready:.1f}% of talent pool</div>\n        </div>\n        <div class="kpi-card">\n            <div class="kpi-label">Senior Talent (8+ yrs)</div>\n            <div class="kpi-value">{senior_count:,}</div>\n            <div class="kpi-sub">Experienced / Leadership tier</div>\n        </div>\n    </div>\n    ', unsafe_allow_html=True)

def render_podium_card(candidate: pd.Series, rank: int):
    configs = {1: ('podium-1', 'badge-gold', '🥇 Rank #1 · Premier Applicant'), 2: ('podium-2', 'badge-silver', '🥈 Rank #2 · High-Potential'), 3: ('podium-3', 'badge-bronze', '🥉 Rank #3 · Strong Contender')}
    card_cls, badge_cls, rank_text = configs.get(rank, ('podium-3', 'badge-bronze', f'Rank #{rank}'))
    cid = candidate.get('enrollee_id', 'N/A')
    score = candidate.get('composite_score', 0.0)
    intent = candidate.get('progression_score', 0.0) * 100
    exp = candidate.get('experience', 'N/A')
    hrs = candidate.get('training_hours', 0)
    badge = candidate.get('badge', 'Qualified Applicant')
    pills = ''.join([f'<span class="pill-tag">{h}</span>' for h in candidate.get('highlights', [])])
    st.markdown(f'\n    <div class="podium-card {card_cls}">\n        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.6rem;">\n            <span class="{badge_cls}">{rank_text}</span>\n            <span style="font-size: 1.6rem; font-weight: 800; color: #fff;">{score:.1f}%</span>\n        </div>\n        <div style="font-size: 1.1rem; font-weight: 700; color: #fff;">Candidate #{cid}</div>\n        <div style="font-size: 0.8rem; color: #a5b4fc; font-weight: 600; margin-bottom: 0.5rem;">✨ {badge}</div>\n        <div style="display: flex; gap: 1rem; background: rgba(0,0,0,0.25); padding: 0.5rem; border-radius: 8px; font-size: 0.8rem;">\n            <div>Intent: <b style="color: #34d399;">{intent:.1f}%</b></div>\n            <div>Exp: <b>{exp} yrs</b></div>\n            <div>Training: <b>{hrs} hrs</b></div>\n        </div>\n        <div style="margin-top: 0.4rem;">{pills}</div>\n    </div>\n    ', unsafe_allow_html=True)

def render_candidate_360(cand: pd.Series):
    cid = cand.get('enrollee_id', 'N/A')
    rank = cand.get('rank', '-')
    match = cand.get('composite_score', 0.0)
    intent = cand.get('progression_score', 0.0) * 100.0
    status = cand.get('status', 'N/A')
    badge = cand.get('badge', 'Qualified Applicant')
    with st.container(border=True):
        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown(f'### Candidate #{cid} &nbsp; `Rank #{rank}` &nbsp; `{status}`')
            st.caption(f'✨ Talent Tier: **{badge}**')
        with col2:
            sub1, sub2 = st.columns(2)
            with sub1:
                st.metric('Composite Match', f'{match:.1f}%')
            with sub2:
                st.metric('Cohort Standing', f"Top {max(0.1, 100.0 - cand.get('percentile', 0.0)):.1f}%")
        st.divider()
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric('⚡ Transition Intent', f'{intent:.1f}%')
        with k2:
            st.metric('💼 Seniority Depth', f"{clean_val(cand.get('experience'))} yrs")
        with k3:
            st.metric('📚 Upskilling Hours', f"{cand.get('training_hours', 0)} hrs")
        with k4:
            cdi = cand.get('city_development_index', 0.0)
            st.metric('🏙️ City Dev Index', f'{cdi:.3f}' if isinstance(cdi, (int, float)) else str(cdi))
        st.divider()
        c_a, c_b, c_c = st.columns(3)
        with c_a:
            st.markdown('**🎓 Academic Profile**')
            st.write(f"• **Education Level:** {clean_val(cand.get('education_level'))}")
            st.write(f"• **Major Discipline:** {clean_val(cand.get('major_discipline'))}")
            st.write(f"• **University:** {clean_val(cand.get('enrolled_university'))}")
            st.write(f"• **Gender:** {clean_val(cand.get('gender'))}")
        with c_b:
            st.markdown('**💼 Work Experience**')
            st.write(f"• **Experience:** {clean_val(cand.get('experience'))} years")
            st.write(f"• **Relevance:** {clean_val(cand.get('relevent_experience'))}")
            st.write(f"• **Company Size:** {clean_val(cand.get('company_size'))}")
            st.write(f"• **Company Type:** {clean_val(cand.get('company_type'))}")
        with c_c:
            st.markdown('**📍 Transition Indicators**')
            st.write(f"• **Last Job Change:** {clean_val(cand.get('last_new_job'))}")
            st.write(f"• **Location Code:** {clean_val(cand.get('city'))}")
            st.write(f"• **Training Hours:** {cand.get('training_hours', 0)} hrs")
            rel_pct = cand.get('score_relevance', 0.5) * 100
            st.write(f'• **Domain Relevance Score:** {rel_pct:.0f}%')
        highlights = cand.get('highlights', [])
        if highlights:
            st.markdown('---')
            st.markdown('**Key Strengths:** ' + ' '.join([f"<span class='pill-tag'>{h}</span>" for h in highlights]), unsafe_allow_html=True)

def plot_talent_radar(top_candidates: pd.DataFrame) -> go.Figure:
    categories = ['Intent', 'Experience', 'Education', 'Training', 'Relevance']
    fig = go.Figure()
    colors = ['#f59e0b', '#94a3b8', '#f97316', '#6366f1', '#06b6d4']
    for i, (_, row) in enumerate(top_candidates.head(5).iterrows()):
        vals = [row.get('score_intent', 0) * 100, row.get('score_experience', 0) * 100, row.get('score_education', 0) * 100, row.get('score_training', 0) * 100, row.get('score_relevance', 0) * 100]
        vals.append(vals[0])
        fig.add_trace(go.Scatterpolar(r=vals, theta=categories + [categories[0]], fill='toself', name=f"#{row.get('recommendation_rank', i + 1)} (ID: {row.get('enrollee_id')})", line=dict(color=colors[i % len(colors)], width=2), opacity=0.4))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100], showticklabels=False, gridcolor='rgba(255,255,255,0.08)'), angularaxis=dict(gridcolor='rgba(255,255,255,0.08)', linecolor='rgba(255,255,255,0.1)'), bgcolor='rgba(0,0,0,0)'), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=40, r=40, t=30, b=30), height=380, legend=dict(orientation='h', yanchor='bottom', y=-0.25, xanchor='center', x=0.5, font=dict(size=11, color='#cbd5e1')))
    return fig

def main():
    artifacts = load_artifacts()
    col_brand, col_status = st.columns([3, 1])
    with col_brand:
        st.markdown("<h1 style='margin-bottom: 0;'>⚡ TalentIQ <span style='font-size: 1.1rem; color: #818cf8;'>Candidate Intelligence Platform</span></h1>", unsafe_allow_html=True)
        st.caption('AI-powered job-change prediction, 360° candidate profiles, and multi-criteria talent shortlisting.')
    with col_status:
        if artifacts:
            st.markdown(f"<div style='text-align: right; padding-top: 10px;'><span class='pill-tag' style='background: rgba(16, 185, 129, 0.15); color: #34d399;'>● Model Active: {artifacts['best_model_name']}</span></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='text-align: right; padding-top: 10px;'><span class='pill-tag' style='background: rgba(239, 68, 68, 0.15); color: #f87171;'>● Model Offline</span></div>", unsafe_allow_html=True)
    if not artifacts:
        st.error('Model artifacts not found in `Trained_Model/`. Run `python train_model.py` first.')
        st.stop()
    with st.sidebar:
        st.markdown('### 📂 Applicant Pool')
        data_source = st.selectbox('Select Candidate Cohort', ['Built-in Benchmark (aug_test.csv)', 'Full Training Cohort (aug_train.csv)', 'Upload Custom CSV'], index=0)
        raw_df = None
        if 'Custom' in data_source:
            uploaded = st.file_uploader('Upload CSV', type=['csv'])
            if uploaded:
                raw_df = pd.read_csv(uploaded)
        elif 'Training' in data_source:
            p = DATA_DIR / '1_Raw_Data' / 'aug_train.csv'
            if not p.exists():
                p = DATA_DIR / 'aug_train.csv'
            if p.exists():
                raw_df = pd.read_csv(p)
        else:
            p = DATA_DIR / '1_Raw_Data' / 'aug_test.csv'
            if not p.exists():
                p = DATA_DIR / 'aug_test.csv'
            if p.exists():
                raw_df = pd.read_csv(p)
        if raw_df is None:
            st.info('Awaiting applicant dataset...')
            st.stop()
        st.markdown('---')
        st.markdown('### 🎯 Ranking Strategy')
        strategy_key = st.selectbox('Recruiter Strategy Preset', ['balanced', 'senior', 'fast_hire', 'high_growth'], format_func=lambda x: {'balanced': 'Balanced Fit (Recommended)', 'senior': 'Senior Leadership & Depth', 'fast_hire': 'Immediate Transition / Fast Hire', 'high_growth': 'High-Velocity Upskillers'}.get(x, x))
        preset = STRATEGY_PRESETS[strategy_key]
        with st.expander('⚙️ Fine-Tune Weights'):
            custom_weights = {'intent': st.slider('Transition Intent', 0.0, 1.0, preset['intent'], 0.05), 'experience': st.slider('Experience Seniority', 0.0, 1.0, preset['experience'], 0.05), 'education': st.slider('Education Attainment', 0.0, 1.0, preset['education'], 0.05), 'training': st.slider('Upskilling Hours', 0.0, 1.0, preset['training'], 0.05), 'relevance': st.slider('Domain Relevance', 0.0, 1.0, preset['relevance'], 0.05)}
        st.markdown('---')
        st.markdown('### 🔍 Filters')
        sel_edu = st.multiselect('Education', ['Primary School', 'High School', 'Graduate', 'Masters', 'Phd'], default=[])
        min_exp, max_exp = st.slider('Experience (years)', 0, 22, (0, 22))
        only_ready = st.checkbox('Only Transition Ready (Prob ≥ 50%)', value=False)
    inferred_df = run_inference(raw_df, artifacts)
    engine = CandidateRankingEngine()
    ranked_df = engine.score_candidates(inferred_df, strategy=strategy_key, custom_weights=custom_weights)
    filtered_df = ranked_df.copy()
    if sel_edu and 'education_level' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['education_level'].isin(sel_edu)]
    if 'exp_years' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['exp_years'].between(min_exp, max_exp)]
    if only_ready:
        filtered_df = filtered_df[filtered_df['prediction'] == 1]
    top_10 = filtered_df.head(10).copy()
    top_10['recommendation_rank'] = range(1, len(top_10) + 1)
    total_candidates = len(filtered_df)
    top10_avg = top_10['composite_score'].mean() if len(top_10) > 0 else 0.0
    ready_count = int((filtered_df['prediction'] == 1).sum())
    senior_count = int((filtered_df.get('exp_years', pd.Series(0)) >= 8).sum())
    render_kpi_bar(total_candidates, top10_avg, ready_count, senior_count)
    tab_top10, tab_pool, tab_shap, tab_model = st.tabs(['🏆 Top 10 Recommended Applicants', '👥 Full Talent Pool & All Details', '🔬 Applicant Deep-Dive & SHAP', '📊 Model Health & Governance'])
    with tab_top10:
        st.markdown('### 🏆 Top 10 Recommended Applicants')
        st.caption('Multi-criteria shortlist synthesizing transition probability, seniority, degree level, and upskilling commitment.')
        if len(top_10) == 0:
            st.warning('No candidates match current filters.')
        else:
            col_p1, col_p2, col_p3 = st.columns(3)
            with col_p1:
                if len(top_10) >= 1:
                    render_podium_card(top_10.iloc[0], 1)
            with col_p2:
                if len(top_10) >= 2:
                    render_podium_card(top_10.iloc[1], 2)
            with col_p3:
                if len(top_10) >= 3:
                    render_podium_card(top_10.iloc[2], 3)
            col_table, col_radar = st.columns([3, 2])
            with col_table:
                st.markdown('#### 📋 Leaderboard')
                cols_to_show = ['recommendation_rank', 'enrollee_id', 'composite_score', 'progression_score', 'percentile', 'experience', 'education_level', 'training_hours', 'badge']
                show_cols = [c for c in cols_to_show if c in top_10.columns]
                renames = {'recommendation_rank': 'Rank', 'enrollee_id': 'Candidate ID', 'composite_score': 'Match Score (%)', 'progression_score': 'Intent (%)', 'percentile': 'Cohort Top (%)', 'experience': 'Experience', 'education_level': 'Education', 'training_hours': 'Training (hrs)', 'badge': 'Badge'}
                disp_df = top_10[show_cols].rename(columns=renames)
                st.dataframe(disp_df.style.format({'Match Score (%)': '{:.1f}%', 'Intent (%)': '{:.1%}', 'Cohort Top (%)': '{:.1f}%'}).background_gradient(subset=['Match Score (%)'], cmap='Purples'), use_container_width=True, height=360)
                st.download_button('📥 Export Top 10 Candidates (CSV)', data=top_10.to_csv(index=False), file_name='top_10_recommended_applicants.csv', mime='text/csv')
            with col_radar:
                st.markdown('#### 🕸️ Competency Radar Comparison')
                st.plotly_chart(plot_talent_radar(top_10), use_container_width=True)
    with tab_pool:
        st.markdown(f'### 👥 Full Talent Pool ({len(filtered_df):,} Candidates)')
        with st.expander('🔍 360° Candidate Profile Inspector', expanded=False):
            ids = filtered_df['enrollee_id'].dropna().astype(int).tolist() if 'enrollee_id' in filtered_df.columns else []
            if ids:
                sel_id = st.selectbox('Select Candidate ID to inspect:', ids, index=0)
                render_candidate_360(filtered_df[filtered_df['enrollee_id'] == sel_id].iloc[0])
        col_l, col_r = st.columns(2)
        with col_l:
            fig_h = px.histogram(filtered_df, x='composite_score', nbins=25, title='Match Score Distribution (%)', color_discrete_sequence=['#6366f1'])
            fig_h.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#94a3b8'), height=240, margin=dict(t=35, b=20, l=20, r=20))
            st.plotly_chart(fig_h, use_container_width=True)
        with col_r:
            status_df = filtered_df['status'].value_counts().reset_index()
            status_df.columns = ['status', 'count']
            fig_d = px.pie(status_df, names='status', values='count', hole=0.55, title='Transition Intent Breakdown', color='status', color_discrete_map={'Ready for Transition': '#34d399', 'Stable in Role': '#64748b'})
            fig_d.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#94a3b8'), height=240, margin=dict(t=35, b=20, l=20, r=20))
            st.plotly_chart(fig_d, use_container_width=True)
        st.markdown('#### 📑 Applicants Roster')
        c_srch, c_lim = st.columns([3, 1])
        with c_srch:
            query = st.text_input('Filter applicants by ID, City, Education, or Discipline:', placeholder='e.g. 10856, STEM, Masters, city_103')
        with c_lim:
            lim_choice = st.selectbox('Display Count', [f'All ({len(filtered_df):,})', 'Top 100', 'Top 500', 'Top 1,000'], index=0)
        pool_view = filtered_df.copy()
        if query.strip():
            q = query.strip().lower()
            mask = False
            for c in ['enrollee_id', 'city', 'education_level', 'major_discipline', 'badge']:
                if c in pool_view.columns:
                    mask = mask | pool_view[c].astype(str).str.lower().str.contains(q)
            pool_view = pool_view[mask]
        if '100' in lim_choice:
            pool_view = pool_view.head(100)
        elif '500' in lim_choice:
            pool_view = pool_view.head(500)
        elif '1,000' in lim_choice:
            pool_view = pool_view.head(1000)
        cols_roster = [c for c in ['rank', 'enrollee_id', 'composite_score', 'progression_score', 'percentile', 'status', 'badge', 'experience', 'education_level', 'major_discipline', 'training_hours', 'relevent_experience', 'company_size', 'city'] if c in pool_view.columns]
        st.dataframe(pool_view[cols_roster], use_container_width=True, height=450, hide_index=True)
        st.download_button('📥 Download Full Candidate Dataset (CSV)', data=filtered_df[cols_roster].to_csv(index=False), file_name='all_candidates.csv', mime='text/csv')
    with tab_shap:
        st.markdown('### 🔬 Explainable AI · Candidate Diagnostic (SHAP)')
        st.caption('Inspect how specific candidate features pushed their job-change prediction up or down.')
        c_ids = filtered_df['enrollee_id'].dropna().astype(int).tolist() if 'enrollee_id' in filtered_df.columns else []
        if not c_ids:
            st.info('No candidates to inspect.')
        else:
            chosen_id = st.selectbox('Candidate to Inspect:', c_ids[:200], format_func=lambda x: f'Candidate #{x}')
            c_row = filtered_df[filtered_df['enrollee_id'] == chosen_id].iloc[0]
            render_candidate_360(c_row)
            c_rec, c_shap_plot = st.columns([1, 2])
            with c_rec:
                st.markdown('#### 💡 Recruiter Talking Points')
                score_val = c_row.get('progression_score', 0.0)
                if score_val >= 0.65:
                    st.info('🔥 **High Urgency**: Strong likelihood of actively interviewing. Schedule outreach immediately.')
                elif score_val >= 0.5:
                    st.info('⚡ **Open to Transition**: Meets interest threshold. Focus discussion on specific career growth.')
                else:
                    st.info('🛡️ **Stable in Current Role**: Low immediate change drive. Highlight compensation and leadership incentives.')
            with c_shap_plot:
                explainer = load_shap_explainer(artifacts['best_model'], np.array(list(filtered_df['_X'].values)))
                if explainer:
                    try:
                        X_val = np.array(c_row['_X'])
                        feats = c_row['_feats']
                        sv = explainer.shap_values(X_val.reshape(1, -1))
                        vals = sv[1][0] if isinstance(sv, list) and len(sv) > 1 else sv[0] if sv.ndim == 2 else sv
                        sdf = pd.DataFrame({'feature': feats, 'impact': vals})
                        sdf = sdf.reindex(sdf['impact'].abs().sort_values().index).tail(10)
                        colors = ['#f87171' if v > 0 else '#60a5fa' for v in sdf['impact']]
                        fig_bar = go.Figure(go.Bar(x=sdf['impact'], y=sdf['feature'], orientation='h', marker_color=colors, text=sdf['impact'].round(3), textposition='outside'))
                        fig_bar.update_layout(title='Feature Impact on Transition (SHAP Values)', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#94a3b8'), xaxis=dict(gridcolor='rgba(255,255,255,0.05)', title='+ drives change, - favors staying'), yaxis=dict(gridcolor='rgba(255,255,255,0.05)'), height=350, margin=dict(l=10, r=30, t=40, b=30))
                        st.plotly_chart(fig_bar, use_container_width=True)
                    except Exception as err:
                        st.warning(f'Could not compute SHAP plot: {err}')
                else:
                    st.info('SHAP explainer loading...')
    with tab_model:
        st.markdown('### 📊 Model Governance & Evaluation')
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown(f"**Champion Model:** `{artifacts['best_model_name']}`")
            st.markdown(f"**Total Features Selected:** `{len(artifacts['selected_features'])}`")
            metrics = artifacts.get('metrics', [])
            if metrics:
                st.table(pd.DataFrame(metrics)[['model', 'auc_roc', 'f1', 'precision', 'recall']])
        with col_m2:
            st.markdown('**Selected Feature Set:**')
            st.write(', '.join([f'`{f}`' for f in artifacts['selected_features']]))
if __name__ == '__main__':
    main()
