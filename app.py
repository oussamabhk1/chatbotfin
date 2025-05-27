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

# === CUSTOM CSS FOR BANKING STYLE + RTL SUPPORT ===
st.markdown(
    """
    <style>
        body {
            background-color: #eef2f7;
            color: #333;
        }
        .stApp {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .chat-container {
            max-width: 800px;
            margin: auto;
            padding: 20px;
            border-radius: 10px;
            background-color: #ffffff;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        }
        .header {
            text-align: center;
            padding-bottom: 10px;
            border-bottom: 1px solid #ddd;
        }
        .header h1 {
            color: #007bff;
            font-size: 2em;
            margin: 0;
        }
        .header h2 {
            color: #555;
            font-size: 1.2em;
            margin-top: 5px;
        }
        .footer {
            margin-top: 40px;
            text-align: center;
            font-size: 0.9em;
            color: #aaa;
        }

        /* Chat Bubbles */
        .user-bubble {
            background-color: #dcf8c6;
            align-self: flex-end;
            border-radius: 10px;
            padding: 10px;
            margin: 5px 0;
            max-width: 80%;
            align-items: flex-end;
            float: right;
            clear: both;
            text-align: right;
        }
        .bot-bubble {
            background-color: #e1f5fe;
            align-self: flex-start;
            border-radius: 10px;
            padding: 10px;
            margin: 5px 0;
            max-width: 80%;
            align-items: flex-start;
            float: left;
            clear: both;
            text-align: left;
        }
        .rtl {
            direction: rtl;
            text-align: right !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

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
        nn_models['ar'] = NearestNeighbors(n_neighbors=1, metric="cosine").fit(embeddings['ar'])
    else:
        st.warning("⚠️ Colonnes arabes absentes ou vides dans le CSV.")

    return embeddings, nn_models

embeddings, nn_models = build_embeddings(df)

# === EXTRACTION VIREMENT SETUP ===
client = Groq(api_key="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")  # Replace securely in production

def encode_image_file(uploaded_file):
    return base64.b64encode(uploaded_file.read()).decode("utf-8")

def extract_invoice_data(base64_image):
    system_prompt = """
    Extract payment data and return JSON with these exact fields:
    - payer: {name: string, account: string (8 digits)}
    - payee: {name: string, account: string (20 digits)}
    - date: string (format DD/MM/YYYY)
    - amount_words: string (French)
    - reason: string
    Return null for missing fields. Maintain this structure exactly.
    """

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract all invoice data"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                    ]
                }
            ],
            temperature=0.0,
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        st.error(f"❌ Erreur lors de l'extraction de l'image : {str(e)}")
        return {}

def convert_french_amount(words):
    french_numbers = {
        'zero': 0, 'un': 1, 'deux': 2, 'trois': 3, 'quatre': 4,
        'cinq': 5, 'six': 6, 'sept': 7, 'huit': 8, 'neuf': 9,
        'dix': 10, 'onze': 11, 'douze': 12, 'treize': 13,
        'quatorze': 14, 'quinze': 15, 'seize': 16,
        'dix-sept': 17, 'dix-huit': 18, 'dix-neuf': 19,
        'vingt': 20, 'trente': 30, 'quarante': 40,
        'cinquante': 50, 'soixante': 60, 'soixante-dix': 70,
        'quatre-vingt': 80, 'quatre-vingt-dix': 90,
        'cent': 100, 'cents': 100, 'mille': 1000
    }

    words = words.lower().replace('dinars', '').replace('dinar', '').strip()
    total = current = 0
    for word in words.split():
        if word in french_numbers:
            val = french_numbers[word]
            if val >= 100:
                current = 1 if current == 0 else current
                total += current * val
                current = 0
            else:
                current += val
    return total + current

def validate_date(date_str):
    try:
        datetime.strptime(date_str, "%d/%m/%Y")
        return True
    except ValueError:
        return False

def validate_invoice_fields(data):
    results = []
    results.append("✅ Payer name" if data.get('payer', {}).get('name') else "❌ Missing payer name")
    results.append("✅ Payee name" if data.get('payee', {}).get('name') else "❌ Missing payee name")
    results.append("✅ Payer account" if data.get('payer', {}).get('account') and len(data.get('payer', {}).get('account', '')) == 8 else "❌ Invalid payer account")
    results.append("✅ Payee account" if data.get('payee', {}).get('account') and len(data.get('payee', {}).get('account', '')) == 20 else "❌ Invalid payee account")
    results.append("✅ Valid date" if validate_date(data.get('date', '')) else "❌ Invalid or missing date")
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

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# File Upload Section (MOVED FIRST)
uploaded_file = st.file_uploader("📎 Télécharger un virement à analyser (.png/.jpg)", type=["png", "jpg", "jpeg"])

# Handle File Upload
if uploaded_file:
    base64_img = encode_image_file(uploaded_file)
    st.markdown(f'<div class="user-bubble">📎 Fichier uploadé</div>', unsafe_allow_html=True)
    st.image(uploaded_file, caption="Virement reçu", use_column_width=True)

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

        st.markdown(f'<div class="bot-bubble">{result}</div>', unsafe_allow_html=True)
        st.session_state.chat_history.append({"role": "bot", "label": "🤖 BankMate", "content": result})

# User Input Section (MOVED AFTER FILE UPLOAD)
user_input = st.text_input("💬 Posez votre question :", placeholder="Exemple: Comment consulter mon solde ?")

# Handle User Input
if user_input:
    st.session_state.chat_history.append({"role": "user", "label": "👤 Vous", "content": user_input})
    
    try:
        lang = detect(user_input)
    except LangDetectException:
        lang = 'en'

    # Fallback logic
    if lang not in ['en', 'fr', 'ar']:
        lang = 'en'
    elif lang == 'ar' and ('Answer_ar' not in df.columns or df['Answer_ar'].isnull().all()):
        st.warning("⚠️ Données arabes indisponibles, basculement vers l'anglais.")
        lang = 'en'

    query = model.encode(user_input)
    distances, indices = nn_models[lang].kneighbors([query])
    idx = indices[0][0]

    profile_col = f"Profile_{lang}" if lang != "en" else "Profile"
    answer_col = f"Answer_{lang}" if lang != "en" else "Answer"
    response_text = df.iloc[idx][answer_col]
    profile_text = df.iloc[idx][profile_col]

    if lang == 'ar':
        st.markdown(
            f"""
            <div class="bot-bubble rtl">
                <strong>🤖 BankMate:</strong><br>
                <b>الملف المعني:</b> <i>{profile_text}</i><br>
                <b>الرد:</b> {response_text}
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="bot-bubble"><strong>🤖 BankMate:</strong><br>'
            f'🔍 Profil concerné: <i>{profile_text}</i><br>'
            f'📌 Réponse: {response_text}</div>',
            unsafe_allow_html=True,
        )

    st.session_state.chat_history.append({
        "role": "bot",
        "label": "🤖 BankMate",
        "content": (
            f"<strong>🔍 Profil concerné:</strong> <i>{profile_text}</i><br>"
            f"<strong>📌 Réponse:</strong> {response_text}"
        ) if lang != 'ar' else (
            f"<strong>الملف المعني:</strong> <i>{profile_text}</i><br>"
            f"<strong>الرد:</strong> {response_text}"
        )
    })

# Show chat history
for msg in st.session_state.chat_history:
    bubble_class = "user-bubble" if msg["role"] == "user" else "bot-bubble"
    content = msg["content"]
    st.markdown(f'<div class="{bubble_class}"><strong>{msg["label"]}:</strong><br>{content}</div>', unsafe_allow_html=True)

# Close chat container
st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown('<div class="footer">© 2025 BankMate - Tous droits réservés.</div>', unsafe_allow_html=True)
