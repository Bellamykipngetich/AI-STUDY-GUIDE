"""
app.py
Streamlit web application for the Exam Question Predictor.

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import tempfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wordcloud import WordCloud

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Exam Question Predictor",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1E3A8A, #3B82F6);
        padding: 2rem; border-radius: 12px; color: white;
        text-align: center; margin-bottom: 2rem;
    }
    .metric-card {
        background: #F8FAFC; border: 1px solid #E2E8F0;
        border-radius: 8px; padding: 1rem; text-align: center;
    }
    .priority-high   { background: #FEE2E2; border-left: 4px solid #EF4444; padding: 0.5rem 1rem; border-radius: 4px; margin: 4px 0; }
    .priority-medium { background: #FEF9C3; border-left: 4px solid #EAB308; padding: 0.5rem 1rem; border-radius: 4px; margin: 4px 0; }
    .priority-low    { background: #DCFCE7; border-left: 4px solid #22C55E; padding: 0.5rem 1rem; border-radius: 4px; margin: 4px 0; }
    .stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ─── Helpers ─────────────────────────────────────────────────────────────────

@st.cache_data
def load_data(csv_path: str):
    return pd.read_csv(csv_path)

@st.cache_resource
def load_model_artifacts():
    from utils.features import load_artifacts
    import joblib
    try:
        vectorizer, lda, dictionary, label_encoder = load_artifacts()
        scaler = joblib.load("models/scaler.pkl")
        model  = joblib.load("models/best_model.pkl")
        return vectorizer, lda, dictionary, label_encoder, scaler, model
    except Exception as e:
        return None

def make_wordcloud(text: str):
    wc = WordCloud(width=700, height=300, background_color='white',
                   colormap='Blues', max_words=60).generate(text)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    plt.tight_layout()
    return fig

def priority_color(p: str):
    if 'High'    in p: return '🔴'
    if 'Medium'  in p: return '🟡'
    if 'Emerging' in p: return '🟢'
    return '⚪'

# ─── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/thumb/5/5d/Kabarak_University_seal.png/200px-Kabarak_University_seal.png",
                 width=120, use_column_width=False)
st.sidebar.title("📚 Exam Question Predictor")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate", [
    "🏠 Home",
    "📊 Topic Analysis",
    "🔮 Predict Exam Topics",
    "📄 Analyze Your Paper",
    "📈 Model Performance"
])

st.sidebar.markdown("---")
st.sidebar.info("Upload your past paper PDFs in `data/raw/` using the naming format:\n\n`SubjectCode_Year_Semester.pdf`\n\nThen run `python train.py`")

# ─── Check if data/models exist ──────────────────────────────────────────────
DATA_CSV    = "data/processed/questions.csv"
MODEL_READY = os.path.exists("models/best_model.pkl")
DATA_READY  = os.path.exists(DATA_CSV)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ═══════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("""
    <div class="main-header">
        <h1>📚 Exam Question Predictor</h1>
        <p style="font-size:1.1rem; opacity:0.9;">AI-powered past paper analysis & topic prediction for smarter exam prep</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    if DATA_READY:
        df = load_data(DATA_CSV)
        with col1:
            st.metric("Total Questions", len(df))
        with col2:
            st.metric("Subjects", df['subject'].nunique())
        with col3:
            st.metric("Years Covered", f"{int(df['year'].min())}–{int(df['year'].max())}" if df['year'].max() > 0 else "N/A")
        with col4:
            st.metric("Papers Processed", df['filename'].nunique())
    else:
        for col, label, val in zip([col1, col2, col3, col4],
                                    ["Total Questions", "Subjects", "Years Covered", "Papers Processed"],
                                    ["—", "—", "—", "—"]):
            with col:
                st.metric(label, val)

    st.markdown("---")
    st.subheader("How it works")
    steps = [
        ("1️⃣ Collect Past Papers", "Download PDF exam papers from your university repository and place them in `data/raw/`."),
        ("2️⃣ Run the Pipeline", "Execute `python train.py` to extract questions, run EDA, and train the prediction model."),
        ("3️⃣ Analyze Topics", "Explore topic frequencies, trends, and keyword analysis per subject."),
        ("4️⃣ Get Predictions", "See which topics are High Priority for your next exam based on historical patterns."),
    ]
    cols = st.columns(4)
    for col, (title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"**{title}**")
            st.caption(desc)

    st.markdown("---")
    if not DATA_READY:
        st.warning("⚠️ No processed data found. Add PDF files to `data/raw/` and run `python train.py` to get started.")
    elif not MODEL_READY:
        st.warning("⚠️ Data found but no trained model. Run `python train.py` to train the model.")
    else:
        st.success("✅ Data and model are ready! Use the sidebar to explore.")

# ═══════════════════════════════════════════════════════════════════════════
# PAGE: TOPIC ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📊 Topic Analysis":
    st.title("📊 Topic Frequency Analysis")

    if not DATA_READY:
        st.warning("No data available. Run `python train.py` first.")
        st.stop()

    df = load_data(DATA_CSV)
    subjects = ['All Subjects'] + sorted(df['subject'].unique().tolist())
    selected_subject = st.selectbox("Select Subject", subjects)

    filtered = df if selected_subject == 'All Subjects' else df[df['subject'] == selected_subject]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Question Type Distribution")
        type_counts = filtered['question_type'].value_counts()
        fig, ax = plt.subplots(figsize=(5, 4))
        colors = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6']
        ax.pie(type_counts, labels=type_counts.index, autopct='%1.1f%%',
               colors=colors[:len(type_counts)], startangle=140)
        ax.set_title('Question Types')
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Questions per Year")
        if df['year'].max() > 0:
            year_counts = filtered.groupby('year').size().reset_index(name='count')
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            ax2.bar(year_counts['year'].astype(str), year_counts['count'], color='steelblue')
            ax2.set_xlabel('Year')
            ax2.set_ylabel('Questions')
            ax2.set_title('Questions per Year')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()
        else:
            st.info("Year data not available in current dataset.")

    st.subheader("🌐 Word Cloud")
    text = " ".join(filtered['clean_question'].dropna())
    if text.strip():
        st.pyplot(make_wordcloud(text))
    else:
        st.info("Not enough text to generate word cloud.")

    st.subheader("📋 Dataset Sample")
    st.dataframe(filtered[['subject', 'year', 'question_type', 'word_count', 'raw_question']].head(20),
                 use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT EXAM TOPICS
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict Exam Topics":
    st.title("🔮 Predict High-Probability Exam Topics")

    if not DATA_READY:
        st.warning("No data available. Run `python train.py` first.")
        st.stop()

    df = load_data(DATA_CSV)

    if 'dominant_topic' not in df.columns:
        st.warning("Topic modeling data not found. Please re-run `python train.py`.")
        st.stop()

    subjects = sorted(df['subject'].unique().tolist())
    selected_subject = st.selectbox("Select Subject", subjects)
    subj_df = df[df['subject'] == selected_subject]

    from utils.predictor import get_topic_frequency, compute_topic_trends, generate_study_plan

    freq_df = get_topic_frequency(subj_df, subject=selected_subject)
    trends_df = compute_topic_trends(freq_df)

    st.subheader(f"📌 Topic Priority Rankings — {selected_subject}")

    high    = trends_df[trends_df['priority'].str.startswith('🔴')]
    medium  = trends_df[trends_df['priority'].str.startswith('🟡')]
    emerging = trends_df[trends_df['priority'].str.startswith('🟢')]
    low     = trends_df[trends_df['priority'].str.startswith('⚪')]

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("🔴 High Priority", len(high))
    with col2: st.metric("🟡 Medium Priority", len(medium))
    with col3: st.metric("🟢 Emerging", len(emerging))
    with col4: st.metric("⚪ Low Priority", len(low))

    st.markdown("---")
    if not high.empty:
        st.markdown("### 🔴 High Priority Topics (study first)")
        for _, row in high.iterrows():
            st.markdown(f"""<div class="priority-high">
                <strong>Topic {row['dominant_topic']}</strong> — Avg Frequency: {row['avg_frequency']}% | 
                Trend: {'↑ Rising' if row['trend_slope'] > 0 else '↓ Declining'} | 
                Seen in {row['years_seen']} year(s)
            </div>""", unsafe_allow_html=True)

    if not medium.empty:
        st.markdown("### 🟡 Medium Priority Topics")
        for _, row in medium.iterrows():
            st.markdown(f"""<div class="priority-medium">
                <strong>Topic {row['dominant_topic']}</strong> — Avg Frequency: {row['avg_frequency']}% | 
                Trend: {'↑ Rising' if row['trend_slope'] > 0 else '↓ Declining'}
            </div>""", unsafe_allow_html=True)

    if not emerging.empty:
        st.markdown("### 🟢 Emerging Topics (new / rising)")
        for _, row in emerging.iterrows():
            st.markdown(f"""<div class="priority-low">
                <strong>Topic {row['dominant_topic']}</strong> — Avg Frequency: {row['avg_frequency']}% | Trend: ↑ Rising
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📚 Recommended Study Time Allocation")
    alloc_data = {
        'Priority Level': ['🔴 High Priority', '🟡 Medium Priority', '🟢 Emerging Topics', '⚪ Low Priority'],
        'Recommended Time': ['40–50%', '30–35%', '10–15%', '5% or less'],
        'Topics in this category': [len(high), len(medium), len(emerging), len(low)]
    }
    st.table(pd.DataFrame(alloc_data))

# ═══════════════════════════════════════════════════════════════════════════
# PAGE: ANALYZE YOUR PAPER
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📄 Analyze Your Paper":
    st.title("📄 Analyze a New Exam Paper")
    st.markdown("Upload a past paper PDF and instantly see which topics it covers and their priority levels.")

    if not MODEL_READY:
        st.warning("No trained model found. Run `python train.py` first.")
        st.stop()

    uploaded = st.file_uploader("Upload Exam Paper (PDF)", type=['pdf'])
    subject_hint = st.text_input("Subject Code (optional, e.g. COMP322)", value="")

    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        with st.spinner("Extracting and analyzing questions..."):
            from utils.preprocessor import extract_text_from_pdf, segment_questions, clean_text
            from utils.predictor import predict_topics_for_paper, compute_topic_trends, get_topic_frequency
            from utils.features import load_artifacts

            raw_text = extract_text_from_pdf(tmp_path)
            raw_questions = segment_questions(raw_text)
            cleaned = [clean_text(q) for q in raw_questions]
            cleaned = [c for c in cleaned if len(c.split()) >= 3]

        if not cleaned:
            st.error("Could not extract readable questions from this PDF. Try another file.")
        else:
            st.success(f"✅ Extracted {len(cleaned)} questions from the uploaded paper.")

            try:
                vectorizer, lda, dictionary, label_encoder = load_artifacts()

                # Load trends from existing data if available
                trends_df = None
                if DATA_READY:
                    df = load_data(DATA_CSV)
                    if 'dominant_topic' in df.columns:
                        subj_df = df if not subject_hint else df[df['subject'] == subject_hint]
                        freq_df = get_topic_frequency(subj_df)
                        trends_df = compute_topic_trends(freq_df)

                predictions = predict_topics_for_paper(cleaned, lda, dictionary, trends_df)

                st.subheader("📋 Question-Level Predictions")
                st.dataframe(predictions, use_container_width=True)

                st.subheader("📊 Topic Distribution in This Paper")
                topic_counts = predictions['predicted_topic'].value_counts()
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.bar(topic_counts.index, topic_counts.values, color='steelblue')
                ax.set_xlabel('Topic')
                ax.set_ylabel('Number of Questions')
                ax.set_title('Topic Distribution in Uploaded Paper')
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            except Exception as e:
                st.error(f"Could not load model artifacts: {e}. Run `python train.py` first.")

        os.unlink(tmp_path)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE: MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Performance":
    st.title("📈 Model Performance Report")

    results_path = "models/evaluation_results.csv"
    if not os.path.exists(results_path):
        st.warning("No evaluation results found. Run `python train.py` first.")
        st.stop()

    results_df = pd.read_csv(results_path, index_col=0)

    st.subheader("📊 Model Comparison Table")
    st.dataframe(results_df.style.highlight_max(axis=0, color='#DBEAFE'), use_container_width=True)

    st.subheader("📈 Performance Chart")
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    available = [m for m in metrics if m in results_df.columns]
    df_plot = results_df[available].astype(float)
    x = np.arange(len(df_plot))
    width = 0.2
    colors = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444']
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, metric in enumerate(available):
        ax.bar(x + i * width, df_plot[metric], width, label=metric.capitalize(), color=colors[i])
    ax.set_xticks(x + width * (len(available) - 1) / 2)
    ax.set_xticklabels(df_plot.index, rotation=15)
    ax.set_ylim(0, 1.1)
    ax.set_title('Model Performance Comparison', fontsize=13, fontweight='bold')
    ax.set_ylabel('Score')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Show saved plots if available
    for plot_name in ['confusion_matrix', 'feature_importance']:
        plot_files = [f for f in os.listdir('eda_outputs') if f.startswith(plot_name)] \
                     if os.path.exists('eda_outputs') else []
        if plot_files:
            st.subheader(f"{'Confusion Matrix' if 'confusion' in plot_name else 'Feature Importance'}")
            st.image(os.path.join('eda_outputs', plot_files[0]))

cd exam_predictor && git init && git add . && git commit -m "Initial commit: Exam Question Predictor" && git remote add origin https://github.com/Bellamykipngetich/AI-STUDY-GUIDE.git && git branch -M main && git push -u origin main
