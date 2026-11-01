"""MailScope — interactive NLP classification and experiment analysis workspace.

Start: python -m streamlit run dashboard.py
The interface reads the local trained pipeline and produces metrics only from actual
predictions / recorded experiment artifacts; no example scores are hard-coded.
"""
from __future__ import annotations

from html import escape
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from spam_classifier.dashboard_utils import (
    MAX_BATCH_ROWS, classify_batch, evaluate_saved_model,
    explain_linear_prediction, load_metrics, predict_one,
)

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / 'artifacts' / 'spam_pipeline.joblib'
REPORTS = ROOT / 'artifacts' / 'reports'
DATASET = ROOT / 'emails.csv'

st.set_page_config(page_title='MailScope | ML Classification Studio',
                   page_icon='✉️', layout='wide', initial_sidebar_state='collapsed')

st.markdown('''
<style>
:root { --blue:#7795ff; --muted:#8e9db9; --ink:#eef3ff; --panel:#111b2e; --line:#263751; }
.stApp { background:
  radial-gradient(ellipse 750px 460px at 92% -15%, rgba(61,91,215,.22), transparent 78%),
  radial-gradient(ellipse 600px 400px at 0% 25%, rgba(34,85,145,.09), transparent 75%), #090f1e; }
.block-container { max-width:1240px; padding-top:2.15rem; padding-bottom:5rem; }
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarNav"], [data-testid="stAppDeployButton"],
.stAppDeployButton, [data-testid="stToolbar"], [data-testid="stDecoration"],
#MainMenu, footer { display:none!important; }
header[data-testid="stHeader"] { background:transparent!important; height:0!important; }
html,body,[data-testid="stAppViewContainer"] { color:var(--ink); }
.brandline { display:flex; align-items:center; justify-content:space-between; margin:0 0 1.1rem; }
.brand { display:flex; align-items:center; gap:11px; color:#e8edff; font-size:1.04rem;
 font-weight:800; letter-spacing:.075em; }
.brandmark { display:inline-flex;align-items:center;justify-content:center;
 background:linear-gradient(145deg,#5269ed,#9a71ff);border-radius:12px;width:35px;height:35px;
 font-size:20px;box-shadow:0 5px 20px rgba(93,107,233,.28); }
.brandmeta { color:#7f93b7; font-weight:600; letter-spacing:.12em; font-size:.66rem; }
.hero {padding:2.0rem 2.1rem 1.85rem; margin-bottom:1.3rem; border-radius:22px;
 background:linear-gradient(116deg,rgba(29,45,89,.86),rgba(20,32,58,.94) 60%,rgba(25,31,68,.84));
 border:1px solid rgba(123,147,226,.2);box-shadow:0 14px 42px rgba(0,0,0,.16); }
.eyebrow {color:#9db4ff; font-size:.70rem;letter-spacing:.18em;font-weight:800;margin-bottom:.55rem;}
.hero h1 {font-size:clamp(2.0rem,3.8vw,3.15rem);line-height:1.14; letter-spacing:-.048em;
 color:#f5f7ff;margin:0 0 .75rem;font-weight:800;}
.hero p {font-size:1.0rem;color:#a9b8d3;line-height:1.65;margin:0;max-width:800px;}
.pill {display:inline-block;margin-top:1rem;color:#c8d6ff;background:rgba(112,144,250,.12);
 border:1px solid rgba(129,153,235,.25);border-radius:20px;padding:5px 12px;
 font-weight:600;font-size:.73rem;letter-spacing:.02em;}
.sectioneyebrow {font-size:.67rem;letter-spacing:.16em;color:#90a7db;font-weight:800;margin:1rem 0 .2rem;}
.paneltitle {color:#f3f6ff;font-size:1.33rem;font-weight:800;letter-spacing:-.025em;margin:0 0 .35rem;}
.panelsub {color:#91a2c0;font-size:.89rem;line-height:1.6;margin:0 0 1.05rem;}
.statrow {display:flex;flex-wrap:wrap;gap:12px;margin:14px 0 20px;}
.statbox {flex:1;min-width:160px;background:#121f37;border:1px solid #2a3b57;border-radius:15px;padding:17px 20px;}
.statlabel {font-size:.74rem;font-weight:650;color:#91a7cf;text-transform:uppercase;letter-spacing:.075em;}
.statvalue {font-size:1.55rem;color:#f6f7ff;font-weight:800;margin-top:5px;}
.statfoot {font-size:.76rem;color:#8c9dbc;margin-top:2px;}
.resultbox {border-radius:17px;background:linear-gradient(125deg,#172946,#121c31);
 border:1px solid #365077;padding:22px 25px;margin:1rem 0;}
.resultbox.spam {background:linear-gradient(125deg,#432735,#231b2f);border-color:#78465f;}
.resultlabel {font-size:.72rem;color:#a8b8d5;letter-spacing:.12em;font-weight:700;}
.resultvalue {font-size:2.1rem;color:#f4f8ff;font-weight:800;margin:.2rem 0 .4rem;}
.resultsub {font-size:.85rem;color:#a4b7d3;}
.smallnote {color:#8294b1;font-size:.77rem;line-height:1.55;margin-top:.75rem;}
hr {border-color:rgba(137,162,218,.14)!important;}
[data-testid="stTabs"] [data-baseweb="tab-list"] {gap:7px;background:#10192b;padding:7px;
 border:1px solid #263651;border-radius:13px;margin-bottom:1rem;overflow-x:auto;}
[data-testid="stTabs"] button[role="tab"] {border-radius:9px;color:#9cb0cd;
 padding:8px 16px;font-size:.89rem;font-weight:600;white-space:nowrap;}
[data-testid="stTabs"] button[aria-selected="true"] {background:#293c68;color:#f7f9ff!important;}
[data-testid="stTabs"] [data-baseweb="tab-highlight"], [data-testid="stTabs"] [data-baseweb="tab-border"] {display:none;}
[data-testid="stTextArea"] textarea, [data-testid="stSelectbox"] >div>div,
[data-testid="stFileUploader"], [data-testid="stDataFrame"],
[data-testid="stExpander"] {border-radius:12px!important;}
div.stButton button[kind="primary"] {background:linear-gradient(105deg,#5577f6,#7659ee);
 border:0;border-radius:11px;font-weight:700;min-height:44px;box-shadow:0 4px 18px rgba(87,105,238,.18);}
div.stButton button[kind="secondary"] {border:1px solid #375178;background:#162542;
 color:#eaf0ff;border-radius:11px;min-height:42px;}
[data-testid="stMetric"] {background:#121e34;border:1px solid #263b57;border-radius:13px;padding:14px 17px;}
[data-testid="stMetricLabel"] {color:#92a7c5;}
[data-testid="stMetricValue"] {color:#f4f6ff;}
[data-testid="stFileUploader"] {border:1px solid #314766;}
</style>
''', unsafe_allow_html=True)


@st.cache_resource(show_spinner='Loading model…')
def load_model(filepath: str, stamp: int):
    # Only load a model saved by this project; joblib is not safe for untrusted files.
    return joblib.load(filepath)


@st.cache_data(show_spinner=False)
def read_saved_report(filepath: str, stamp: int) -> dict:
    return load_metrics(filepath)


def stat_card(label: str, value: str, foot: str = '') -> str:
    # Arguments are escaped because this is rendered as HTML.
    return (f'<div class="statbox"><div class="statlabel">{escape(label)}</div>'
            f'<div class="statvalue">{escape(value)}</div>'
            f'<div class="statfoot">{escape(foot)}</div></div>')


def matrix_plot(cm: list[list[int]]):
    fig, ax = plt.subplots(figsize=(5.6,4.1), facecolor='#111b2e')
    ax.set_facecolor('#111b2e')
    arr = np.asarray(cm)
    heat = ax.imshow(arr, cmap='Blues', vmin=0, vmax=max(1, arr.max()))
    for i in range(2):
        for j in range(2):
            # Choose a contrasting number color using the actual cell background.
            r, g, b, _ = heat.cmap(heat.norm(int(arr[i, j])))
            luminance = .2126*r + .7152*g + .0722*b
            number_color = '#f8faff' if luminance < .52 else '#14243b'
            ax.text(j, i, f'{arr[i,j]:,}', ha='center', va='center', fontsize=16,
                    fontweight='bold', color=number_color)
    ax.set_xticks([0,1],labels=['Not Spam','Spam'],color='#b6c5e0')
    ax.set_yticks([0,1],labels=['Not Spam','Spam'],color='#b6c5e0')
    ax.set_xlabel('Predicted label', color='#aabbd6', labelpad=9)
    ax.set_ylabel('Actual label', color='#aabbd6', labelpad=9)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return fig


def line_plot(x, y, xlabel, ylabel, *, baseline=False):
    fig, ax = plt.subplots(figsize=(5.6,3.7), facecolor='#111b2e')
    ax.set_facecolor('#111b2e')
    ax.plot(x,y,color='#7795ff',linewidth=2.5)
    if baseline:
        ax.plot([0,1],[0,1],'--',color='#71819c',alpha=.7)
    ax.set(xlim=(0,1),ylim=(0,1))
    ax.set_xlabel(xlabel,color='#aabbd6',labelpad=8)
    ax.set_ylabel(ylabel,color='#aabbd6',labelpad=8)
    ax.grid(color='#263651',alpha=.65)
    ax.tick_params(colors='#8fa4c4')
    for spine in ax.spines.values():
        spine.set_color('#30405b')
    fig.tight_layout()
    return fig


st.markdown('''<div class="brandline"><div class="brand"><span class="brandmark">✦</span>
MAILSCOPE <span style="font-size:.62rem;color:#8398c1;font-weight:600">/ ML STUDIO</span></div>
<div class="brandmeta">TEXT INTELLIGENCE · PYTHON / SCIKIT-LEARN</div></div>
<div class="hero"><div class="eyebrow">NATURAL LANGUAGE PROCESSING</div>
<h1>Email intelligence,<br>made measurable.</h1>
<p>Explore a trained spam-classification model, analyze text predictions, verify held-out performance,
 and inspect the signals behind each decision.</p>
<span class="pill">TF-IDF + SCIKIT-LEARN / Cross-validated model selection</span></div>''',
            unsafe_allow_html=True)

if not MODEL.exists():
    st.error('Trained model missing. Run `python -m spam_classifier.train --data emails.csv --out artifacts` first.')
    st.stop()
try:
    pipeline = load_model(str(MODEL), MODEL.stat().st_mtime_ns)
except Exception as exc:
    st.error(f'Model could not be loaded: {exc}')
    st.stop()

name = type(pipeline.named_steps['classifier']).__name__ if hasattr(pipeline,'named_steps') else 'Trained classifier'
st.markdown('<div class="smallnote">ACTIVE MODEL &nbsp;·&nbsp; '+escape(name)+
            ' &nbsp;·&nbsp; Changes to your dataset or training artifacts are reflected after a new run.</div>',
            unsafe_allow_html=True)

analyze, evaluate, batch, insights = st.tabs([
    '✦  Analyze message', '▥  Model evaluation', '▤  Batch workspace', '◈  Model insights'
])

with analyze:
    st.markdown('<div class="sectioneyebrow">01 / INTERACTIVE INFERENCE</div><div class="paneltitle">Analyze a message</div>'
                '<div class="panelsub">Paste email text or use a sample. Results come from your saved trained model.</div>',
                unsafe_allow_html=True)
    examples = {
        'Custom message': '',
        'Work email': 'Hi team, I attached the revised agenda. Please review it before our Thursday meeting.',
        'Promotional message': 'Congratulations! Claim your cash reward and unlock your free bonus gift today.',
    }
    example = st.selectbox('Example', list(examples), label_visibility='collapsed')
    if st.session_state.get('previous_example') != example:
        st.session_state['message_input'] = examples[example]
        st.session_state.pop('current_result', None)
        st.session_state['previous_example'] = example
    message = st.text_area('Message content',key='message_input',height=188,
                           placeholder='Paste the email you want to classify…',max_chars=20_000)
    if st.button('Analyze message',type='primary',use_container_width=True):
        try:
            st.session_state['current_result'] = predict_one(pipeline,message)
            st.session_state['analyzed_text'] = message
        except ValueError as err:
            st.session_state.pop('current_result',None)
            st.warning(str(err))
    if (st.session_state.get('current_result') is not None
            and st.session_state.get('analyzed_text') == message):
        prediction = st.session_state['current_result']
        is_spam = prediction['prediction'] == 1
        css_name = 'spam' if is_spam else 'ham'
        st.markdown(f'<div class="resultbox {css_name}"><div class="resultlabel">CLASSIFICATION RESULT</div>'
                    f'<div class="resultvalue">{"Spam detected" if is_spam else "Likely legitimate"}</div>'
                    f'<div class="resultsub">Model score for spam class: <b>{prediction["spam_score"]:.1%}</b></div></div>',
                    unsafe_allow_html=True)
        st.progress(float(prediction['spam_score']))
        explanation = explain_linear_prediction(pipeline,message)
        if not explanation.empty:
            st.markdown('<div class="sectioneyebrow">DECISION INSPECTION</div><div class="paneltitle">Contributing terms</div>'
                        '<div class="panelsub">Words and phrases from this message that influenced the classification.</div>',
                        unsafe_allow_html=True)
            values = explanation.copy()
            values['Impact'] = values['Contribution'].abs().round(3)
            st.dataframe(values[['Term','Direction','Impact']],hide_index=True,use_container_width=True)

with evaluate:
    st.markdown('<div class="sectioneyebrow">02 / EXPERIMENT VALIDATION</div><div class="paneltitle">Holdout performance</div>'
                '<div class="panelsub">Re-run the saved model on its original unseen test partition. '
                'The metrics and charts below are computed from predictions when you run the evaluation.</div>',
                unsafe_allow_html=True)
    report_path = REPORTS/'metrics.json'
    report = None
    if report_path.exists():
        try:
            report = read_saved_report(str(report_path),report_path.stat().st_mtime_ns)
        except (ValueError, OSError, KeyError, TypeError) as err:
            st.warning(f'Unable to read training run metadata: {err}')
    if not DATASET.exists():
        st.info('The labeled dataset `emails.csv` is required to calculate fresh holdout metrics.')
    if st.button('Run holdout evaluation',type='primary',disabled=not DATASET.exists(),use_container_width=True):
        try:
            seed = int(report['split']['seed']) if report else 42
            size = float(report['split']['test_size']) if report else .2
            with st.spinner('Classifying held-out messages and calculating scores…'):
                st.session_state['evaluation_result'] = evaluate_saved_model(
                    pipeline, DATASET,seed=seed,test_size=size)
                st.session_state['evaluation_model_version'] = MODEL.stat().st_mtime_ns
        except (ValueError, OSError, KeyError, TypeError) as err:
            st.error(f'Evaluation could not complete: {err}')
    result = st.session_state.get('evaluation_result')
    if result and st.session_state.get('evaluation_model_version') == MODEL.stat().st_mtime_ns:
        m = result['metrics']; cm = result['confusion_matrix']
        st.markdown('<div class="sectioneyebrow">CURRENT EVALUATION RESULTS</div>',unsafe_allow_html=True)
        st.markdown('<div class="statrow">'+
                    stat_card('Accuracy',f'{m["accuracy"]:.2%}',f'{result["sample_count"]:,} messages')+
                    stat_card('Spam precision',f'{m["precision"]:.2%}','Positive-class precision')+
                    stat_card('Spam recall',f'{m["recall"]:.2%}','Positive-class recall')+
                    stat_card('Spam F1',f'{m["f1"]:.2%}','Harmonic mean')+'</div>',unsafe_allow_html=True)
        st.markdown('<div class="paneltitle">Error breakdown</div><div class="panelsub">Actual vs. predicted labels on the held-out test data.</div>',unsafe_allow_html=True)
        left,right = st.columns([1,1])
        with left:
            fig = matrix_plot(cm)
            st.pyplot(fig,use_container_width=True)
            plt.close(fig)
        with right:
            st.markdown('<div class="statrow">'+
                        stat_card('True negatives',str(cm[0][0]),'Correctly classified ham')+
                        stat_card('False positives',str(cm[0][1]),'Ham flagged as spam')+
                        stat_card('False negatives',str(cm[1][0]),'Spam missed')+
                        stat_card('True positives',str(cm[1][1]),'Correctly flagged spam')+'</div>',unsafe_allow_html=True)
            st.markdown(f'<div class="smallnote">Evaluated on {result["sample_count"]:,} held-out messages '
                        f'from {result["dataset_rows"]:,} cleaned records · split seed {result["seed"]}.</div>',unsafe_allow_html=True)
        st.markdown('<div class="paneltitle">Classification curves</div>',unsafe_allow_html=True)
        c1,c2 = st.columns(2)
        with c1:
            fig=line_plot(result['pr_curve']['recall'],result['pr_curve']['precision'],
                          'Recall','Precision')
            st.pyplot(fig,use_container_width=True)
            plt.close(fig)
            st.caption(f'Average precision: {m["average_precision"]:.4f}')
        with c2:
            fig=line_plot(result['roc_curve']['fpr'],result['roc_curve']['tpr'],
                          'False-positive rate','True-positive rate',baseline=True)
            st.pyplot(fig,use_container_width=True)
            plt.close(fig)
            st.caption(f'ROC-AUC: {m["roc_auc"]:.4f}')
        if not result['errors'].empty:
            with st.expander(f'Inspect {len(result["errors"])} misclassified rows'):
                st.dataframe(result['errors'],hide_index=True,use_container_width=True)
        if report and report.get('cv_comparison'):
            st.markdown('<div class="sectioneyebrow">SAVED TRAINING EXPERIMENT</div><div class="paneltitle">Model comparison</div>'
                        '<div class="panelsub">Cross-validation scores recorded when the saved model was trained. '
                        'These are not generated by the holdout button.</div>',unsafe_allow_html=True)
            cv=pd.DataFrame(report['cv_comparison'])
            names={'logistic_regression':'Logistic Regression','multinomial_nb':'Multinomial NB',
                   'bernoulli_nb':'Bernoulli NB'}
            cv['Model']=cv['model'].map(names).fillna(cv['model'])
            cv['CV spam F1']=cv['cv_f1_mean']
            # Horizontal bars keep full model names readable and make scores comparable.
            ordered = cv.sort_values('CV spam F1')
            fig, ax = plt.subplots(figsize=(8.4, 2.7), facecolor='#111b2e')
            ax.set_facecolor('#111b2e')
            bars = ax.barh(ordered['Model'], ordered['CV spam F1'],
                           color='#7896ff', height=.56)
            ax.set_xlim(0, 1.0)
            ax.set_xlabel('Mean cross-validated spam F1', color='#aabbd6', labelpad=9)
            ax.tick_params(colors='#b6c5e0', length=0)
            ax.grid(axis='x', color='#263651', alpha=.5)
            ax.set_axisbelow(True)
            for bar, value in zip(bars, ordered['CV spam F1']):
                ax.text(min(float(value) + .012, .982),
                        bar.get_y() + bar.get_height()/2,
                        f'{value:.4f}', color='#f0f4ff', va='center',
                        ha='left' if value < .97 else 'right', fontsize=10, weight='bold')
            for spine in ax.spines.values():
                spine.set_visible(False)
            fig.tight_layout(pad=1.2)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.dataframe(cv[['Model','cv_f1_mean','cv_f1_std']].rename(
                columns={'cv_f1_mean':'Mean spam F1','cv_f1_std':'Std. deviation'}),
                hide_index=True,use_container_width=True)
    else:
        st.markdown('<div class="smallnote">Select <b>Run holdout evaluation</b> to generate your '
                    'performance dashboard from the actual dataset and saved model.</div>',unsafe_allow_html=True)

with batch:
    st.markdown('<div class="sectioneyebrow">03 / DATASET INFERENCE</div><div class="paneltitle">Classify in batches</div>'
                f'<div class="panelsub">Upload a CSV with a <code>text</code> column (up to {MAX_BATCH_ROWS} messages). '
                'Review predictions and export the results.</div>',unsafe_allow_html=True)
    upload = st.file_uploader('Upload messages CSV',type=['csv'])
    if upload is not None:
        try:
            if upload.size > 2*1024*1024:
                raise ValueError('The CSV exceeds the 2 MB upload limit.')
            df = pd.read_csv(upload)
            st.caption(f'{len(df):,} messages loaded')
            st.dataframe(df.head(5),hide_index=True,use_container_width=True)
            if st.button('Classify uploaded messages',type='primary'):
                st.session_state['batch_predictions'] = classify_batch(pipeline,df)
                st.session_state['batch_signature']=(upload.name,upload.size)
        except (ValueError,UnicodeError,pd.errors.ParserError) as err:
            st.error(str(err))
    if (upload is not None and st.session_state.get('batch_signature')==(upload.name,upload.size)
            and 'batch_predictions' in st.session_state):
        pred=st.session_state['batch_predictions']
        counts=pred['prediction'].value_counts()
        st.markdown('<div class="statrow">'+
                    stat_card('Analyzed',str(len(pred)),'Total messages')+
                    stat_card('Spam',str(int(counts.get('Spam',0))),'Predicted spam')+
                    stat_card('Not spam',str(int(counts.get('Not Spam',0))),'Predicted legitimate')+
                    '</div>',unsafe_allow_html=True)
        st.dataframe(pred,hide_index=True,use_container_width=True)
        st.download_button('Export predictions as CSV',data=pred.to_csv(index=False).encode('utf-8'),
                           file_name='mailscope_predictions.csv',mime='text/csv',use_container_width=True)
    st.caption('Try the included `demo_messages.csv` to explore batch classification.')

with insights:
    st.markdown('<div class="sectioneyebrow">04 / MODEL TRANSPARENCY</div><div class="paneltitle">What the model learns</div>'
                '<div class="panelsub">Explore influential features recorded during model training and understand '
                'how evaluation is structured.</div>',unsafe_allow_html=True)
    features = REPORTS/'top_features.csv'
    if features.exists():
        features_df=pd.read_csv(features)
        if {'class','feature','weight'}.issubset(features_df.columns):
            a,b=st.columns(2)
            for column,group,heading in [(a,'spam','Spam-associated features'),
                                          (b,'ham','Legitimate-associated features')]:
                with column:
                    st.markdown(f'**{heading}**')
                    sub=features_df.loc[features_df['class']==group].head(12)
                    st.dataframe(sub[['feature','weight']].rename(
                        columns={'feature':'Term / phrase','weight':'Model weight'}),
                        hide_index=True,use_container_width=True)
        else:
            st.info('Feature report is missing expected columns. Re-run training to refresh it.')
    else:
        st.info('Run training to generate the model feature report.')
    st.markdown('<div class="paneltitle">Evaluation workflow</div>',unsafe_allow_html=True)
    st.markdown('''**01 — Clean and deduplicate.** Validate labeled messages and remove duplicates before the split.

**02 — Hold out 20%.** Reserve a stratified test partition before selecting a model.

**03 — Cross-validate training data.** Fit TF-IDF inside each fold and compare candidate classifiers.

**04 — Select and fit.** Choose the best training-fold F1 and refit the full pipeline on training examples.

**05 — Verify the result.** Evaluate the saved model on the untouched holdout and inspect errors.
''')
