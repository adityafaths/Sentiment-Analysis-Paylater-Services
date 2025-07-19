import streamlit as st
import pandas as pd
import nltk
import re
import string
import joblib
from io import BytesIO
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import networkx as nx
import string
# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Set page config

st.set_page_config(page_title="NLP Pipeline", layout="wide")

# Title
st.title("🛠️ NLP Processing Pipeline")

# Load model dan TF-IDF (cached)
@st.cache_resource
def load_model():
    model = joblib.load("./svm_gridsearch_smote.pkl")
    vectorizer = joblib.load("./tfidf_vectorizer.pkl")
    return model, vectorizer

svm_model, tfidf_vectorizer = load_model()

# Function for preprocessing
def remove_punctuation(text):
    if isinstance(text, str):
        translator = str.maketrans('', '', string.punctuation)
        return text.translate(translator)
    return text
def text_preprocessing(df):
    st.write("⏳ Memproses teks...")
    
    try:
        #cleaning text
        def clean_text(text):
            text = str(text)
            text = re.sub(r'@\w+', '', text)      # Hapus mention
            text = re.sub(r'http\S+', '', text)    # Hapus URL
            text = re.sub(r'#\w+', '', text)       # Hapus hashtag
            text = re.sub(r'\d+', '', text)        # Hapus angka
            text = remove_punctuation(text)        # Hapus tanda baca
            return text.strip()

        df['full_text'] = df['full_text'].apply(clean_text)
        
        #Tokenisasi dan lowercase
        df['full_text'] = df['full_text'].apply(word_tokenize)
        df['full_text'] = df['full_text'].apply(lambda x: [word.lower() for word in x])
        
    # Normalisasi slang word
        try:
            slang_path = "./list_slang_word.txt"
            file = open(slang_path, "r", encoding="utf-8")
            slang_content = file.read()
            file.close()

            dict_slang = eval(slang_content) 
            df['full_text'] = df['full_text'].apply(lambda x: [dict_slang.get(word, word) for word in x])
        except Exception as e:
            st.warning(f"📛 Gagal memuat kamus slang word: {str(e)} (melewati tahap normalisasi)")

        #Stemming
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        df['full_text'] = df['full_text'].apply(lambda x: [stemmer.stem(word) for word in x])
        
        # Konversi list token ke string
        df['full_text'] = df['full_text'].apply(' '.join)
        
        return df[['full_text']]
    
    except Exception as e:
        st.error(f"Error dalam preprocessing: {str(e)}")
        return None
    

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["📁 Text Preprocessing", "🧠 Sentiment Analysis", "📊 Word Cloud & Stats", "📝 Summary & Insights"])

with tab1:
    with st.sidebar:
        st.header("Upload Dataset")
        uploaded_file = st.file_uploader("Choose file", type=["csv", "xlsx", "txt"])

    if uploaded_file:
        file_type = uploaded_file.name.split(".")[-1]

        # Load file
        if file_type == "csv":
            df = pd.read_csv(uploaded_file)
        elif file_type == "xlsx":
            df = pd.read_excel(uploaded_file)
        elif file_type == "txt":
            df = pd.read_csv(uploaded_file, delimiter="\n", header=None, names=["full_text"])
        else:
            st.error("Unsupported file format!")
            st.stop()

        # Pilih kolom jika bukan .txt
        if file_type != "txt":
            text_column = st.selectbox("Pilih kolom teks:", df.columns)
            df_text_only = df[[text_column]].copy()
            df_text_only.columns = ['full_text']
        else:
            df_text_only = df

        st.session_state.df_text_only = df_text_only
        st.dataframe(df_text_only, use_container_width=True)

        if st.button("🔁 Proses Teks"):
            df_processed = text_preprocessing(st.session_state.df_text_only)
            if df_processed is not None:
                st.session_state.df_sisakan = df_processed
                st.success("Preprocessing berhasil dilakukan!")
                st.dataframe(df_processed, use_container_width=True)
            else:
                st.error("Proses preprocessing gagal!")
    else:
        st.info("Silakan unggah file terlebih dahulu untuk memulai.")
        st.session_state.df_text_only = None
        st.session_state.df_sisakan = None


with tab2:
    st.header("Sentiment Analysis for Processed Data")
    
    # Cek apakah data sudah diproses
    if 'df_sisakan' not in st.session_state or st.session_state.df_sisakan is None:
        st.warning("Silakan upload file dan lakukan preprocessing di tab pertama terlebih dahulu!")
        st.stop()  # Menghentikan eksekusi tab ini
    
    # Lanjutkan jika data tersedia
    df_processed = st.session_state.df_sisakan.copy()

    # Konversi teks ke fitur TF-IDF
    text_tfidf = tfidf_vectorizer.transform(df_processed['full_text'])
    df_processed["sentiment"] = svm_model.predict(text_tfidf)
    st.session_state.df_sisakan = df_processed 

    
    # Tampilkan hasil klasifikasi
    st.dataframe(df_processed, use_container_width=True)
    
    # Tombol untuk mengunduh hasil sebagai CSV
    csv = df_processed.to_csv(index=False).encode('utf-8')
    st.download_button(label="📥 Download Hasil Klasifikasi",
                       data=csv,
                       file_name="hasil_sentiment_analysis.csv",
                       mime="text/csv")


with tab3:
    st.header("📊 Word Cloud & Sentiment Distribution")

    if 'df_sisakan' not in st.session_state:
        st.warning("Silakan proses data di tab preprocessing terlebih dahulu.")
        st.stop()

    df_processed = st.session_state.df_sisakan.copy()

    # 1️⃣ Sentiment Distribution (Tanpa Neutral)
    st.subheader("📊 Sentiment Distribution")
    
    # Filter hanya positive dan negative
    sentiment_counts = df_processed["sentiment"].value_counts().reindex(["positive", "negative"], fill_value=0)
    
    # Buat bar chart dengan matplotlib untuk customisasi lebih baik
    fig, ax = plt.subplots(figsize=(8, 4))
    sentiment_counts.plot(kind='bar', 
                         color=['#4CAF50', '#FF5252'], 
                         width=0.6,
                         edgecolor='black',
                         ax=ax)
    
    # Styling plot
    ax.set_title('Distribusi Sentimen', fontsize=14, pad=12, fontweight='bold')
    ax.set_xlabel('Sentimen', fontsize=12, labelpad=10)
    ax.set_ylabel('Jumlah', fontsize=12, labelpad=10)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Tambahkan label nilai di atas bar
    for i, v in enumerate(sentiment_counts):
        ax.text(i, v + 0.5, str(v), 
               ha='center', 
               va='bottom', 
               fontsize=10,
               fontweight='bold')
    
    st.pyplot(fig)

    # 2️⃣ Kumpulkan kata berdasarkan sentimen
    positive_words = []
    negative_words = []

    for _, row in df_processed.iterrows():
        words = row["full_text"].split()
        sentiment = row["sentiment"]
        if sentiment == "positive":
            positive_words.extend(words)
        elif sentiment == "negative":
            negative_words.extend(words)

    # 3️⃣ Word Cloud dengan ukuran lebih besar
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("☁️ Word Cloud - Positive Words")
        if positive_words:
            wordcloud_pos = WordCloud(
                width=1000,  # Diperbesar
                height=600,  # Diperbesar
                background_color="white",
                colormap="Greens",
                max_words=200
            ).generate(" ".join(positive_words))
            
            fig, ax = plt.subplots(figsize=(12, 8))  # Ukuran figure diperbesar
            ax.imshow(wordcloud_pos, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)
    
    with col2:
        st.subheader("☁️ Word Cloud - Negative Words")
        if negative_words:
            wordcloud_neg = WordCloud(
                width=1000,
                height=600,
                background_color="white",
                colormap="Reds",
                max_words=200
            ).generate(" ".join(negative_words))
            
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.imshow(wordcloud_neg, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig) 

    # 4️⃣ 20 Kata Terbanyak dalam Sentimen Positif & Negatif
    st.subheader("🔝 Top 20 Most Frequent Words")

    pos_counts = Counter(positive_words).most_common(20)
    neg_counts = Counter(negative_words).most_common(20)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("✅ Top 20 Positive Words")
        if pos_counts:
            df_pos = pd.DataFrame(pos_counts, columns=["Word", "Count"]).set_index("Word")
            st.bar_chart(df_pos)

    with col4:
        st.subheader("❌ Top 20 Negative Words")
        if neg_counts:
            df_neg = pd.DataFrame(neg_counts, columns=["Word", "Count"]).set_index("Word")
            st.bar_chart(df_neg)
    st.session_state.pos_counts = pos_counts
    st.session_state.neg_counts = neg_counts



 # 5️⃣ Text Network Graph
    col5, col6 = st.columns(2)  
    
    with col5:
        st.subheader("🌐 Text Network Graph (Top 100 Words)")
        
        # Ambil 10 kata teratas dari masing-masing sentimen
        top_pos_words = [word for word, _ in pos_counts[:100]]
        top_neg_words = [word for word, _ in neg_counts[:100]]
        
        # Buat graph dengan parameter yang lebih baik
        G = nx.Graph()
        
        # Tambahkan node dengan atribut visual
        for word in top_pos_words:
            G.add_node(word, 
                      color="#4CAF50",  # Hijau modern
                      size=500 + 30*pos_counts.index((word, Counter(positive_words)[word])))
        
        for word in top_neg_words:
            G.add_node(word, 
                      color="#FF5252",  # Merah modern
                      size=500 + 30*neg_counts.index((word, Counter(negative_words)[word])))
        
        # Tambahkan edge berdasarkan co-occurrence dalam dokumen
        for _, row in df_processed.iterrows():
            words = set(row["full_text"].split())
            for word1 in words:
                for word2 in words:
                    if word1 != word2 and G.has_node(word1) and G.has_node(word2):
                        if G.has_edge(word1, word2):
                            G[word1][word2]["weight"] += 1
                        else:
                            G.add_edge(word1, word2, weight=1)
        
        # Visualisasi yang lebih interaktif dengan kamus posisi
        pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
        
        # Buat figure dengan ukuran yang sesuai
        fig = plt.figure(figsize=(10, 8))
        
        # Gambar edges dengan ketebalan berdasarkan weight
        edges = nx.draw_networkx_edges(
            G, pos,
            edge_color="gray",
            width=[0.5 * G[u][v]['weight'] for u, v in G.edges()],
            alpha=0.4
        )
        
        # Gambar nodes dengan ukuran dan warna
        nodes = nx.draw_networkx_nodes(
            G, pos,
            node_color=[G.nodes[n]['color'] for n in G.nodes],
            node_size=[G.nodes[n]['size'] for n in G.nodes],
            alpha=0.8
        )
        
        # Gambar labels dengan efek bayangan
        labels = nx.draw_networkx_labels(
            G, pos,
            font_size=10,
            font_weight='bold',
            font_family='sans-serif',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, boxstyle='round,pad=0.2')
        )
        
        # Tambahkan efek bayangan pada node
        nodes.set_edgecolor('white')
        nodes.set_linewidth(0.5)
        
        # Styling plot
        plt.title("Word Co-occurrence Network", fontsize=14, pad=20)
        plt.axis('off')
        
        # Tambahkan legenda
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', label='Positive',
                      markerfacecolor='#4CAF50', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label='Negative',
                      markerfacecolor='#FF5252', markersize=10)
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        st.pyplot(fig)
with tab4:
    st.header("📝 Summary & Insights with Sahabat AI")
    
    # Cek apakah data sudah diproses
    if 'df_sisakan' not in st.session_state or st.session_state.df_sisakan is None:
        st.warning("Silakan upload file dan lakukan preprocessing di tab pertama terlebih dahulu!")
        st.stop()
    
    # Inisialisasi model Sahabat AI
    sahabat_ai = ChatOllama(
        base_url="http://localhost:11434",  
        model="hf.co/gmonsoon/gemma2-9b-cpt-sahabatai-v1-instruct-GGUF:Q8_0",
        temperature=0.7,
        max_tokens=1024
    )

    # Inisialisasi session state
    if 'analyses' not in st.session_state:
        st.session_state.analyses = {}

    # Validasi data preprocessing
    if 'df_sisakan' not in st.session_state:
        st.warning("Silakan proses data di tab preprocessing terlebih dahulu.")
        st.stop()

    # Ambil data
    df_processed = st.session_state.df_sisakan
    pos_counts = getattr(st.session_state, 'pos_counts', [])
    neg_counts = getattr(st.session_state, 'neg_counts', [])

    # Ambil 5 kata teratas
    top_pos = [word for word, _ in pos_counts[:5]]
    top_neg = [word for word, _ in neg_counts[:5]]

    # UI Layout
    st.subheader("🔍 Analisis Kata Kunci Teratas")

    # Template prompt analisis
    analysis_prompt = ChatPromptTemplate.from_messages([
        ("system", "Anda adalah analis senior platform financial technology paylater."),
        ("human", "{input}")
    ])

    # Chain untuk analisis
    analysis_chain = analysis_prompt | sahabat_ai

    # Section Positif
    st.markdown("## 🟢 5 Kata Positif Dominan")
    for word in top_pos:
        with st.expander(f"POSITIF: {word}", expanded=False):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("### Konteks Positif")
                contexts = [
                    txt for txt, sent in zip(df_processed['full_text'], df_processed['sentiment'])
                    if (word in txt.split()) and (sent == 'positive')
                ][:5]
                
                if contexts:
                    st.write("Contoh penggunaan positif:")
                    for ctx in contexts:
                        st.write(f"- `{ctx}`")
                    
                    if st.button(f"🧠 Analisis Positif", 
                               key=f"pos_{word}",
                               type='primary'):
                        with st.spinner(f"Menganalisis aspek positif {word}..."):
                            try:
                                prompt_text = f"""Berdasarkan konteks:
                                                {contexts}
                                                
                                                [Instruksi Analisis]
                                                1. Identifikasi penyebab kata '{word}' mengapa terletak di sentimen positif : (1 paragraf)
                                                2. Jelaskan 3 dampak positif untuk stakeholder :
                                                3. Berikan 3 rekomendasi spesifik untuk:
                                                   - Regulator
                                                   - Perusahaan Fintech
                                                   - Masyarakat
                                                [Format]
                                                - Poin-poin spesifik
                                                - Hindari generalisasi
                                                - Fokus pada kata kunci '{word}'
                                                - susupkan beberapa kalimat '{contexts}' ke dalam jawaban agar jawaban relevan
                                                """
                                   
                                
                                response = analysis_chain.invoke({"input": prompt_text})
                                st.session_state.analyses[word] = {
                                    'positive': response.content,
                                    'negative': st.session_state.analyses.get(word, {}).get('negative', '')
                                }
                                
                            except Exception as e:
                                st.error(f"Error: Pastikan Ollama server aktif! {str(e)}")
                
                if st.session_state.analyses.get(word, {}).get('positive'):
                    st.markdown("### 📝 Hasil Analisis Positif")
                    st.write(st.session_state.analyses[word]['positive'])

            with col2:
                if word in top_neg:
                    st.markdown("### ⚠️ Juga Muncul di Negatif")
                    st.warning(f"Kata '{word}' juga termasuk dalam 5 besar negatif")

    # Pemisah visual
    st.markdown("---")

    # Section Negatif
    st.markdown("## 🔴 5 Kata Negatif Dominan")
    for word in top_neg:
        with st.expander(f"NEGATIF: {word}", expanded=False):
            col1, col2 = st.columns([1, 2])
            
            with col2:
                st.markdown("### Konteks Negatif")
                contexts = [
                    txt for txt, sent in zip(df_processed['full_text'], df_processed['sentiment'])
                    if (word in txt.split()) and (sent == 'negative')
                ][:5]
                
                if contexts:
                    st.write("Contoh penggunaan negatif:")
                    for ctx in contexts:
                        st.write(f"- `{ctx}`")
                    
                    if st.button(f"🧠 Analisis Negatif", 
                               key=f"neg_{word}",
                               type='secondary'):
                        with st.spinner(f"Menganalisis aspek negatif {word}..."):
                            try:
                                prompt_text = f"""Berdasarkan konteks:
                                                {contexts}
                                                
                                                [Instruksi Analisis]
                                                1. Identifikasi penyebab kata '{word}' mengapa terletak di sentimen negative: (1 paragraf)
                                                2. Jelaskan 3 dampak negative untuk stakeholder :
                                                3. Berikan solusi konkret untuk:
                                                   - Regulator
                                                   - Perusahaan Fintech
                                                   - Pengguna
                                                [Format]
                                                - Poin-poin spesifik
                                                - Hindari generalisasi
                                                - Fokus pada kata kunci '{word}'
                                                - susupkan beberapa kalimat '{contexts}' ke dalam jawaban agar jawaban relevan
                                                """
                                
                                response = analysis_chain.invoke({"input": prompt_text})
                                st.session_state.analyses[word] = {
                                    'negative': response.content,
                                    'positive': st.session_state.analyses.get(word, {}).get('positive', '')
                                }
                                
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
                
                if st.session_state.analyses.get(word, {}).get('negative'):
                    st.markdown("### 📝 Hasil Analisis Negatif")
                    st.write(st.session_state.analyses[word]['negative'])

            with col1:
                if word in top_pos:
                    st.markdown("### ⚠️ Juga Muncul di Positif")
                    st.warning(f"Kata '{word}' juga termasuk dalam 5 besar positif")

    # Reset All
    st.markdown("---")
    if st.button("♻️ Reset Semua Analisis", type="primary"):
        st.session_state.analyses = {}
        st.rerun()

    st.caption("ℹ️ Analisis menggunakan model Sahabat AI 9B Q8_0 - Hasil tergantung kualitas data dan konteks")