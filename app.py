import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).parent
MODEL_PATH = BASE / 'model' / 'fraud_model.pkl'
SCALER_PATH = BASE / 'model' / 'scaler.pkl'
METRICS_PATH = BASE / 'model' / 'metrics.json'
DEMO_PATH = BASE / 'data' / 'demo_transactions.csv'

st.set_page_config(page_title='Credit Card Fraud Detection', page_icon='💳', layout='wide', initial_sidebar_state='expanded')

# ---------- Styling ----------
st.markdown('''
<style>
[data-testid="stAppViewContainer"] { background: #07111f; }
[data-testid="stHeader"] { background: rgba(0,0,0,0); }
.block-container { max-width: 1200px; padding-top: 2rem; padding-bottom: 3rem; }
.hero { padding: 26px 30px; border: 1px solid #20344d; border-radius: 22px; background: linear-gradient(135deg,#0e1d31,#10283e); margin-bottom: 22px; }
.hero h1 { margin: 0; font-size: 42px; }
.hero p { color:#b8c7d9; font-size:17px; margin:8px 0 0; }
.card { background:#0e1a2a; border:1px solid #20344d; border-radius:18px; padding:20px; }
.result-pass { background:#083c2d; border:1px solid #17c989; border-radius:18px; padding:22px; }
.result-fraud { background:#4a171d; border:1px solid #ff5c69; border-radius:18px; padding:22px; }
.metric { background:#0e1a2a; border:1px solid #20344d; border-radius:16px; padding:18px; text-align:center; }
.metric .value { font-size:28px; font-weight:700; }
.metric .label { color:#9fb0c3; font-size:13px; }
.small { color:#9fb0c3; font-size:13px; }
</style>
''', unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    metrics = json.loads(METRICS_PATH.read_text())
    demo = pd.read_csv(DEMO_PATH)
    return model, scaler, metrics, demo

model, scaler, metrics, demo = load_assets()
features = metrics['feature_names']

if 'history' not in st.session_state:
    st.session_state.history = []

# ---------- Helpers ----------
def transform_input(row_df):
    x = row_df[features].copy()
    x[['Time','Amount']] = scaler.transform(x[['Time','Amount']])
    return x

def predict(row_df):
    x = transform_input(row_df)
    prob = float(model.predict_proba(x)[0,1])
    label = int(prob >= 0.5)
    return label, prob

def risk_label(prob):
    if prob < 0.30:
        return 'LOW', '🟢'
    if prob < 0.70:
        return 'MEDIUM', '🟠'
    return 'HIGH', '🔴'

def add_history(row, label, prob):
    st.session_state.history.insert(0, {
        'Amount': float(row['Amount'].iloc[0]),
        'Fraud Probability': round(prob*100, 2),
        'Result': 'Fraud' if label else 'Genuine',
        'Risk': risk_label(prob)[0]
    })
    st.session_state.history = st.session_state.history[:20]

# ---------- Sidebar ----------
st.sidebar.markdown('## 💳 Fraud Analytics')
page = st.sidebar.radio('Navigation', ['🔍 Fraud Prediction', '📊 Model Analytics', '📋 Prediction History', 'ℹ️ About Project'])
st.sidebar.markdown('---')
st.sidebar.caption('Machine Learning • XGBoost • Streamlit')

# ---------- Header ----------
st.markdown('''<div class="hero"><h1>💳 Credit Card Fraud Detection<br>& Risk Analytics System</h1><p>Predict whether a transaction is genuine or potentially fraudulent using machine learning.</p></div>''', unsafe_allow_html=True)

# ---------- Prediction ----------
if page == '🔍 Fraud Prediction':
    st.subheader('🔍 Transaction Risk Assessment')
    st.write('Use a real transaction from the included demo set, or enter the anonymized model features manually.')

    mode = st.radio('Input mode', ['Quick Demo Transaction', 'Advanced Manual Input'], horizontal=True)

    if mode == 'Quick Demo Transaction':
        idx = st.number_input('Demo transaction number', min_value=1, max_value=len(demo), value=1, step=1)
        selected = demo.iloc[[idx-1]].copy()
        st.info('Demo mode uses an actual row from your uploaded Credit Card Fraud dataset. The Class column is hidden from the prediction input.')
        c1,c2,c3 = st.columns(3)
        c1.metric('Transaction Amount', f"₹{selected['Amount'].iloc[0]:,.2f}")
        c2.metric('Transaction Time', f"{selected['Time'].iloc[0]:,.0f}")
        c3.metric('Dataset Label', 'Fraud' if selected['Class'].iloc[0] == 1 else 'Genuine')
        row = selected.drop(columns=['Class'])

    else:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        c1,c2 = st.columns(2)
        with c1:
            amount = st.number_input('Transaction Amount', min_value=0.0, value=250.0, step=10.0)
        with c2:
            time = st.number_input('Transaction Time', min_value=0.0, value=10000.0, step=100.0)
        with st.expander('Advanced anonymized features (V1–V28)', expanded=False):
            vals = {}
            cols = st.columns(4)
            for i, f in enumerate([f'V{i}' for i in range(1,29)]):
                with cols[i % 4]:
                    vals[f] = st.number_input(f, value=0.0, format='%.6f')
        row = pd.DataFrame([{**{'Time':time, 'Amount':amount}, **vals}])
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('')
    if st.button('🔍 CHECK TRANSACTION', type='primary', use_container_width=True):
        label, prob = predict(row)
        risk, icon = risk_label(prob)
        add_history(row, label, prob)
        st.session_state.last = (label, prob, risk, row.copy())

    if 'last' in st.session_state:
        label, prob, risk, lastrow = st.session_state.last
        st.markdown('---')
        if label == 1:
            st.markdown(f'<div class="result-fraud"><h2>🚨 Potential Fraudulent Transaction</h2><p>The model classifies this transaction as potentially fraudulent.</p><h3>{icon} Risk Level: {risk}</h3></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-pass"><h2>✅ Transaction Likely Genuine</h2><p>The model classifies this transaction as likely genuine.</p><h3>{icon} Risk Level: {risk}</h3></div>', unsafe_allow_html=True)
        a,b,c = st.columns(3)
        a.metric('Fraud Probability', f'{prob*100:.2f}%')
        b.metric('Genuine Probability', f'{(1-prob)*100:.2f}%')
        c.metric('Risk Level', risk)
        st.progress(prob, text=f'Fraud risk score: {prob*100:.2f}%')
        st.caption('The probability is the model score, not a guarantee that a real transaction is fraudulent.')

# ---------- Analytics ----------
elif page == '📊 Model Analytics':
    st.subheader('📊 Model Performance & Dataset Analytics')
    m1,m2,m3,m4,m5 = st.columns(5)
    m1.metric('Accuracy', f"{metrics['accuracy']*100:.2f}%")
    m2.metric('Precision', f"{metrics['precision']*100:.2f}%")
    m3.metric('Recall', f"{metrics['recall']*100:.2f}%")
    m4.metric('F1 Score', f"{metrics['f1']*100:.2f}%")
    m5.metric('ROC-AUC', f"{metrics['roc_auc']:.3f}")

    st.markdown('### Confusion Matrix')
    cm = np.array(metrics['confusion_matrix'])
    cm_df = pd.DataFrame(cm, index=['Actual Genuine','Actual Fraud'], columns=['Predicted Genuine','Predicted Fraud'])
    st.dataframe(cm_df, use_container_width=True)

    c1,c2 = st.columns(2)
    with c1:
        st.markdown('### Dataset Class Distribution')
        class_df = pd.DataFrame({'Class':['Genuine','Fraud'], 'Transactions':[metrics['genuine_count'],metrics['fraud_count']]})
        st.bar_chart(class_df.set_index('Class'))
    with c2:
        st.markdown('### Top Model Features')
        fi = pd.read_csv(BASE/'model'/'feature_importance.csv') if (BASE/'model'/'feature_importance.csv').exists() else None
        if fi is not None:
            st.bar_chart(fi.head(10).set_index('Feature')['Importance'])
        else:
            st.info('Feature importance file not found.')

    st.info(f"Dataset: {metrics['genuine_count'] + metrics['fraud_count']:,} transactions • {metrics['fraud_count']:,} fraud cases • PR-AUC: {metrics['pr_auc']:.3f}")

# ---------- History ----------
elif page == '📋 Prediction History':
    st.subheader('📋 Prediction History')
    if not st.session_state.history:
        st.info('No predictions yet. Go to Fraud Prediction and check a transaction.')
    else:
        st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True, hide_index=True)
        if st.button('Clear History'):
            st.session_state.history = []
            st.rerun()

# ---------- About ----------
else:
    st.subheader('ℹ️ About the Project')
    st.markdown('''
### Objective
Build a machine-learning system that identifies potentially fraudulent credit-card transactions and presents a probability-based risk assessment.

### Machine Learning Pipeline
**Dataset → Preprocessing → Train/Test Split → XGBoost → Evaluation → Saved Model → Streamlit Web App**

### Technologies
- Python
- Pandas & NumPy
- Scikit-learn
- XGBoost
- Joblib
- Streamlit
- Matplotlib/Seaborn-ready analytics

### Dataset
The project uses the uploaded Credit Card Fraud Detection dataset with **284,807 transactions**, including **492 fraud transactions**. The `V1–V28` fields are anonymized/PCA-transformed features.

### Why accuracy is not enough
Fraud detection is an imbalanced classification problem. Therefore, this project reports **Precision, Recall, F1 Score, ROC-AUC and PR-AUC**, along with the confusion matrix.
''')
