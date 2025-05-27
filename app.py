import os
import json
import base64
import pandas as pd
from datetime import datetime

# Suppress Torch warnings from Streamlit watcher
os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "0"

import streamlit as st
from groq import Groq
from langdetect import detect, LangDetectException
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

# === PAGE CONFIGURATION ===
st.set_page_config(page_title="BankMate", layout="centered")

# === CUSTOM CSS FOR BANKING STYLE + DARK MODE ===
st.markdown(
    """
    <style>
        body {
            background-color: #1e1e1e;
            color: #f0f0f0;
            transition: all 0.3s ease;
        }
        .stApp {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .chat-container {
            max-width: 800px;
            margin: auto;
            padding: 20px;
            border-radius: 10px;
            background-color: #2c2c2c;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        }
        .header {
            text-align: center;
            padding-bottom: 10px;
            border-bottom: 1px solid #444;
        }
        .header h1 {
            color: #00aaff;
            font-size: 2em;
            margin: 0;
        }
        .header h2 {
            color: #aaa;
            font-size: 1.2em;
            margin-top: 5px;
        }
        .footer {
            margin-top: 40px;
            text-align: center;
            font-size: 0.9em;
            color: #666;
        }
        .user-bubble {
            background-color: #3b3b3b;
            align-self: flex-end;
            border-radius: 10px;
            padding: 10px;
            margin: 5px 0;
            max-width: 80%;
        }
        .bot-bubble {
            background-color: #37474f;
            align-self: flex-start;
            border-radius: 10px;
            padding: 10px;
            margin: 5px 0;
            max-width: 80%;
        }
        .dark-mode-toggle {
            position: fixed;
            top: 10px;
            right: 10px;
            z-index: 999;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# === STATE MANAGEMENT ===
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# === TOGGLE DARK MODE ===
def toggle_dark_mode():
    st.session_state.dark_mode = not st.session_state.dark_mode

st.markdown('<div class="dark-mode-toggle">', unsafe_allow_html=True)
if st.button("🌙 Basculer en mode sombre" if not st.session_state.dark_mode else "☀️ Basculer en mode clair"):
    toggle_dark_mode()
st.markdown('</div>', unsafe_allow_html=True)

# === HEADER ===
st.markdown(
    """
    <div class="header">
        <h1>🏦 BankMate</h1>
        <h2>Votre assistant bancaire intelligent</h2>
    </div>
    """,
    unsafe_allow_html=True,
)

# Logo
col1, col2, col3 = st.columns([1, 4, 1])
with col2:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712109.png ", width=100)

# === INITIALISATION CHATBOT ===
@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

@st.cache_data
def load_data():
    df = pd.read_csv("cleanedTranslatedBankFAQs.csv", usecols=[
        "Question", "Answer", "Class", "Profile",
        "Profile_fr", "Profile_ar", "Class_fr", "Class_ar",
        "Question_fr", "Question_ar", "Answer_fr", "Answer_ar"
    ])
    return df

model = load_model()
df = load_data()

@st.cache_resource
def build_embeddings(data):
    embeddings = {}
    nn_models = {}

    en_questions = data["Profile"].fillna('') + " - " + data["Question"].fillna('')
    embeddings['en'] = model.encode(en_questions.tolist())
    nn_models['en'] = NearestNeighbors(n_neighbors=1, metric="cosine").fit(embeddings['en'])

    fr_questions = data["Profile_fr"].fillna('') + " - " + data["Question_fr"].fillna('')
    embeddings['fr'] = model.encode(fr_questions.tolist())
    nn_models['fr'] = NearestNeighbors(n_neighbors=1, metric="cosine").fit(embeddings['fr'])

    if "Question_ar" in data.columns and not data["Question_ar"].isnull().all():
        ar_questions = data["Profile_ar"].fillna('') + " - " + data["Question_ar"].fillna('')
        embeddings['ar'] = model.encode(ar_questions.tolist())
        nn_models['ar'] = NearestButtons
        results.append("✅ Reason provided" if data.get('reason') else "❌ Missing reason")
        return results

# === MAIN APP LOGIC ===
now = datetime.now().hour
if now < 12:
    greeting = "☀️ Bonjour !"
elif now < 18:
    greeting = "🌤️ Bon après-midi !"
else:
    greeting = "🌙 Bonsoir !"

st.markdown(f"<p style='text-align:center; font-size:1.2em;'>{greeting} Comment puis-je vous aider aujourd’hui ?</p>", unsafe_allow_html=True)

# Chat container
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Show chat history
for msg in st.session_state.chat_history:
    bubble_class = "user-bubble" if msg["role"] == "user" else "bot-bubble"
    st.markdown(f'<div class="{bubble_class}"><strong>{msg["label"]}:</strong><br>{msg["content"]}</div>', unsafe_allow_html=True)

# User Input Section
user_input = st.text_input("💬 Posez votre question :", placeholder="Exemple: Comment consulter mon solde ?")

# File Upload Section
uploaded_file = st.file_uploader("📎 Télécharger un virement à analyser (.png/.jpg)", type=["png", "jpg", "jpeg"])

# Clear Chat Button
if st.button("🧹 Effacer la conversation"):
    st.session_state.chat_history = []
    st.experimental_rerun()

# Handle User Input
if user_input:
    st.session_state.chat_history.append({"role": "user", "label": "👤 Vous", "content": user_input})
    
    try:
        lang = detect(user_input)
    except LangDetectException:
        lang = 'en'

    if lang not in ['en', 'fr']:
        lang = 'en'

    query = model.encode(user_input)
    distances, indices = nn_models[lang].kneighbors([query])
    idx = indices[0][0]

    profile_col = f"Profile_{lang}" if lang != "en" else "Profile"
    answer_col = f"Answer_{lang}" if lang != "en" else "Answer"

    bot_response = (
        f'🔍 Profil concerné: <i>{df.iloc[idx][profile_col]}</i><br>'
        f'📌 Réponse: {df.iloc[idx][answer_col]}'
    )

    st.session_state.chat_history.append({"role": "bot", "label": "🤖 BankMate", "content": bot_response})
    st.experimental_rerun()

# Handle File Upload
if uploaded_file:
    st.session_state.chat_history.append({"role": "user", "label": "📎 Fichier uploadé", "content": ""})
    base64_img = encode_image_file(uploaded_file)

    with st.spinner("🧠 Analyse du virement en cours..."):
        extracted_data = extract_invoice_data(base64_img)

        result = (
            f'📄 Données extraites :<br>'
            f'👤 Payer: {extracted_data.get("payer", {}).get("name", "")} ({extracted_data.get("payer", {}).get("account", "")})<br>'
            f'👤 Payee: {extracted_data.get("payee", {}).get("name", "")} ({extracted_data.get("payee", {}).get("account", "")})<br>'
            f'📅 Date: {extracted_data.get("date", "")}<br>'
            f'💬 Raison: {extracted_data.get("reason", "")}<br>'
            f'💶 Montant (lettres): {extracted_data.get("amount_words", "")}<br><br>'
            f'✅ Validation:<br>' +
            "<br>".join([f"- {check}" for check in validate_invoice_fields(extracted_data)])
        )
        st.session_state.chat_history.append({"role": "bot", "label": "🤖 BankMate", "content": result})
    st.experimental_rerun()

# Close chat container
st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown('<div class="footer">© 2025 BankMate - Tous droits réservés.</div>', unsafe_allow_html=True)
