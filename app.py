# app.py
import streamlit as st
import pandas as pd
import joblib
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="Mobile JKN Sentiment Analysis",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk styling modern
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .prediction-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .stTextArea > div > div > textarea {
        border-radius: 10px;
        border: 2px solid #667eea;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Load model dan vectorizer
@st.cache_resource
def load_models():
    try:
        model = joblib.load('model/svm_model.pkl')
        vectorizer = joblib.load('model/vectorizer.pkl')
        return model, vectorizer
    except FileNotFoundError:
        st.error("❌ Model atau vectorizer tidak ditemukan. Pastikan file ada di folder 'model/'")
        return None, None

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/hasil_preprocessing_mobile_jkn.csv')
        df['stemming'] = df['stemming'].apply(lambda x: ' '.join(ast.literal_eval(x)) if isinstance(x, str) else x)
        return df
    except FileNotFoundError:
        st.error("❌ Dataset tidak ditemukan. Pastikan file ada di folder 'data/'")
        return None

# Header utama
st.markdown('<h1 class="main-header">📱 Mobile JKN Sentiment Analysis</h1>', unsafe_allow_html=True)

# Load data dan model
model, vectorizer = load_models()
df = load_data()

# Sidebar dengan styling modern
st.sidebar.markdown("""
<div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 1rem;">
    <h2 style="color: white; margin: 0;">🚀 Navigation</h2>
    <p style="color: white; margin: 0; opacity: 0.8;">Pilih menu untuk analisis</p>
</div>
""", unsafe_allow_html=True)

option = st.sidebar.radio(
    "📋 Menu Utama",
    ["🏠 Dashboard", "📄 Dataset", "🧠 Prediksi Manual", "📈 Visualisasi Sentimen"],
    index=0
)

# Dashboard Overview
if option == "🏠 Dashboard":
        # Quick visualization
        st.markdown("### 📊 Distribusi Sentimen Cepat")
        if 'label' in df.columns:
            fig = px.pie(
                values=df['label'].value_counts().values, 
                names=df['label'].value_counts().index,
                title="Distribusi Sentimen Mobile JKN",
                color_discrete_sequence=['#ff6b6b', '#4ecdc4', '#45b7d1']
            )
            fig.update_layout(
                title_font_size=20,
                font=dict(size=14),
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)

# Tampilan dataset
elif option == "📄 Dataset":
    st.markdown("## 📄 Dataset Mobile JKN")
    
    if df is not None:
        # Info dataset
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📋 Informasi Dataset")
            st.info(f"📊 **Total baris:** {len(df)} | 📈 **Total kolom:** {len(df.columns)}")
        
        with col2:
            st.markdown("### 🔍 Filter Data")
            show_rows = st.selectbox("Tampilkan baris:", [10, 20, 50, 100, len(df)])
        
        # Display dataset dengan styling
        st.markdown("### 📊 Data Preview")
        st.dataframe(
            df.head(show_rows), 
            use_container_width=True,
            height=400
        )
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Dataset",
            data=csv,
            file_name='mobile_jkn_dataset.csv',
            mime='text/csv'
        )

# Prediksi manual
elif option == "🧠 Prediksi Manual":
    st.markdown("## 🧠 Prediksi Sentimen Manual")
    
    if model is not None and vectorizer is not None:
        # Form prediksi dengan styling modern
        st.markdown("### ✍️ Masukkan Ulasan Anda")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            user_input = st.text_area(
                "Tulis ulasan tentang aplikasi Mobile JKN:",
                placeholder="Contoh: Aplikasi Mobile JKN sangat membantu untuk mengecek BPJS...",
                height=150
            )
        
        with col2:
            st.markdown("### 💡 Tips")
            st.info("Tulis ulasan yang jelas dan lengkap untuk hasil prediksi yang lebih akurat!")
        
        if st.button("🔮 Prediksi Sentimen", use_container_width=True):
            if user_input.strip():
                try:
                    # Prediksi
                    input_vector = vectorizer.transform([user_input])
                    hasil_prediksi = model.predict(input_vector)[0]
                    confidence = model.decision_function(input_vector)[0]
                    
                    # Styling hasil prediksi
                    if hasil_prediksi.lower() == 'positif':
                        emoji = "😊"
                        color = "#4CAF50"
                    elif hasil_prediksi.lower() == 'negatif':
                        emoji = "😞"
                        color = "#F44336"
                    else:
                        emoji = "😐"
                        color = "#FF9800"
                    
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, {color}20, {color}40); 
                                padding: 2rem; border-radius: 15px; text-align: center; 
                                border-left: 5px solid {color}; margin: 2rem 0;">
                        <h2 style="color: {color}; margin-bottom: 1rem;">
                            {emoji} Hasil Prediksi: {hasil_prediksi.upper()}
                        </h2>
                        <p style="font-size: 1.1rem; color: #333;">
                            Tingkat kepercayaan: <strong>{abs(confidence):.2f}</strong>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"❌ Terjadi kesalahan dalam prediksi: {str(e)}")
            else:
                st.warning("⚠️ Silakan masukkan teks ulasan terlebih dahulu!")

# Visualisasi hasil
elif option == "📈 Visualisasi Sentimen":
    st.markdown("## 📈 Visualisasi Hasil Sentimen")
    
    if df is not None and 'label' in df.columns:
        # Tabs untuk berbagai visualisasi
        tab1, tab2, tab3 = st.tabs(["📊 Distribusi Sentimen", "☁️ Word Cloud", "📈 Analisis Detail"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                # Bar chart dengan Plotly
                sentiment_count = df['label'].value_counts()
                fig_bar = px.bar(
                    x=sentiment_count.index, 
                    y=sentiment_count.values,
                    title="📊 Distribusi Sentimen (Bar Chart)",
                    labels={'x': 'Label Sentimen', 'y': 'Jumlah'},
                    color=sentiment_count.values,
                    color_continuous_scale=['#ff6b6b', '#4ecdc4', '#45b7d1']
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            
            with col2:
                # Pie chart dengan Plotly
                fig_pie = px.pie(
                    values=sentiment_count.values, 
                    names=sentiment_count.index,
                    title="🥧 Distribusi Sentimen (Pie Chart)",
                    color_discrete_sequence=['#ff6b6b', '#4ecdc4', '#45b7d1']
                )
                st.plotly_chart(fig_pie, use_container_width=True)
        
        with tab2:
            st.markdown("### ☁️ Word Cloud Analysis")
            
            # Load dan tampilkan gambar wordcloud
            try:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("#### 😊 Word Cloud Positif")
                    img_positive = Image.open('images/positif.png')
                    st.image(img_positive, use_container_width=True)
                
                with col2:
                    st.markdown("#### 😞 Word Cloud Negatif")
                    img_negative = Image.open('images/negatif.png')
                    st.image(img_negative, use_container_width=True)
                
                with col3:
                    st.markdown("#### 😐 Word Cloud Keseluruhan")
                    img_neutral = Image.open('images/seluruh kata.png')
                    st.image(img_neutral, use_container_width=True)
                    
            except FileNotFoundError:
                st.error("❌ Gambar word cloud tidak ditemukan. Pastikan file ada di folder 'images/'")
                st.info("📁 File yang dibutuhkan: wordcloud_positif.png, wordcloud_negatif.png, wordcloud_netral.png")
        
        with tab3:
            st.markdown("### 📈 Analisis Detail")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Statistik detail
                st.markdown("#### 📊 Statistik Sentimen")
                total_data = len(df)
                for sentiment in df['label'].unique():
                    count = len(df[df['label'] == sentiment])
                    percentage = (count / total_data) * 100
                    st.metric(
                        label=f"{sentiment.title()}",
                        value=f"{count}",
                        delta=f"{percentage:.1f}%"
                    )
            
            with col2:
                # Donut chart
                fig_donut = go.Figure(data=[go.Pie(
                    labels=sentiment_count.index, 
                    values=sentiment_count.values, 
                    hole=.3
                )])
                fig_donut.update_layout(
                    title="🍩 Donut Chart Sentimen",
                    annotations=[dict(text='Sentimen', x=0.5, y=0.5, font_size=20, showarrow=False)]
                )
                st.plotly_chart(fig_donut, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            border-radius: 10px; color: white; margin-top: 2rem;">
    <h4>🚀 Mobile JKN Sentiment Analysis Dashboard</h4>
    <p>Dikembangkan dengan ❤️ menggunakan Streamlit & Machine Learning</p>
</div>
""", unsafe_allow_html=True)
