import os
import json
import base64
import pandas as pd
from datetime import datetime

# Suppress PyTorch warning in Streamlit (optional)
os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"

import streamlit as st
from groq import Groq
from langdetect import detect, LangDetectException
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

# === PAGE CONFIGURATION ===
st.set_page_config(page_title="Chatbot Bancaire", layout="wide")

# === CUSTOM CSS FOR BANKING STYLE ===
st.markdown(
    """
    <style>
        body {
            background-color: #f5f5f5;
        }
        .stApp {
            font-family: 'Arial', sans-serif;
        }
        .stTabs > div[data-baseweb="tab"] > button {
            color: #333;
            background-color: #f0f0f0;
            border: none;
            padding: 10px 20px;
            margin-right: 10px;
            border-radius: 5px;
            transition: all 0.3s ease;
        }
        .stTabs > div[data-baseweb="tab"][data-selected] > button {
            background-color: #007bff;
            color: white;
        }
        .stButton > button {
            background-color: #007bff;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .stButton > button:hover {
            background-color: #0056b3;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header with Logo
col1, col2 = st.columns([1, 4])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712109.png ", width=60)
with col2:
    st.title("💬 Chatbot Bancaire")
    st.subheader("Votre assistant bancaire intelligent")

# Tabs
tab1, tab2 = st.tabs(["📩 Chatbot Bancaire", "📤 Extraction Virements"])

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

    # English
    en_questions = data["Profile"].fillna('') + " - " + data["Question"].fillna('')
    embeddings['en'] = model.encode(en_questions.tolist())
    nn_models['en'] = NearestNeighbors(n_neighbors=1, metric="cosine").fit(embeddings['en'])

    # French
    fr_questions = data["Profile_fr"].fillna('') + " - " + data["Question_fr"].fillna('')
    embeddings['fr'] = model.encode(fr_questions.tolist())
    nn_models['fr'] = NearestNeighbors(n_neighbors=1, metric="cosine").fit(embeddings['fr'])

    # Arabic (only if columns exist and are not empty)
    if "Question_ar" in data.columns and not data["Question_ar"].isnull().all():
        ar_questions = data["Profile_ar"].fillna('') + " - " + data["Question_ar"].fillna('')
        embeddings['ar'] = model.encode(ar_questions.tolist())
        nn_models['ar'] = NearestNeighbors(n_neighbors=1, metric="cosine").fit(embeddings['ar'])
    else:
        st.warning("⚠️ Colonnes arabes absentes ou vides dans le CSV.")

    return embeddings, nn_models

embeddings, nn_models = build_embeddings(df)

# === EXTRACTION VIREMENT SETUP ===
# Use environment variable or secrets.toml for production
client = Groq(api_key=os.getenv("GROQ_API_KEY"))  # Set this in secrets or env

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
    results.append("✅ Payer name" if data['payer']['name'] else "❌ Missing payer name")
    results.append("✅ Payee name" if data['payee']['name'] else "❌ Missing payee name")
    results.append("✅ Payer account" if data['payer']['account'] and len(data['payer']['account']) == 8 else "❌ Invalid payer account")
    results.append("✅ Payee account" if data['payee']['account'] and len(data['payee']['account']) == 20 else "❌ Invalid payee account")
    results.append("✅ Valid date" if validate_date(data['date']) else "❌ Invalid or missing date")
    results.append("✅ Reason provided" if data['reason'] else "❌ Missing reason")
    return results

# === TAB 1: CHATBOT BANCAIRE ===
with tab1:
    now = datetime.now().hour
    if now < 12:
        greeting = "☀️ Bonjour !"
    elif now < 18:
        greeting = "🌤️ Bon après-midi !"
    else:
        greeting = "🌙 Bonsoir !"

    st.markdown(f"### {greeting} Comment puis-je vous aider aujourd’hui ?")

    user_input = st.text_input(
        "💡 Posez une question bancaire ci-dessous :",
        placeholder="Exemple: Comment effectuer un virement ?"
    )

    if user_input:
        try:
            lang = detect(user_input)
        except LangDetectException:
            lang = 'en'

        if lang not in ['en', 'fr', 'ar']:
            lang = 'en'

        query = model.encode(user_input)
        distances, indices = nn_models[lang].kneighbors([query])
        idx = indices[0][0]

        profile_col = f"Profile_{lang}" if lang != "en" else "Profile"
        answer_col = f"Answer_{lang}" if lang != "en" else "Answer"

        st.write("### 🏷️ Profil concerné :")
        st.info(df.iloc[idx][profile_col])

        st.write("### 📌 Réponse suggérée :")
        st.success(df.iloc[idx][answer_col])

    # FAQ Section
    st.subheader("🔍 Questions Fréquentes")
    faqs = [
        {"question": "Comment effectuer un virement ?", "answer": "Suivez ces étapes pour effectuer un virement."},
        {"question": "Où trouver mon solde ?", "answer": "Consultez votre compte en ligne."}
    ]
    for faq in faqs:
        with st.expander(faq["question"]):
            st.write(faq["answer"])

# === TAB 2: EXTRACTION VIREMENTS ===
with tab2:
    st.subheader("Uploader un virement à analyser")
    uploaded_file = st.file_uploader("📎 Déposez une image (.png/.jpg)", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        base64_img = encode_image_file(uploaded_file)
        st.image(uploaded_file, caption="Virement uploadée", use_column_width=True)
        with st.spinner("🧠 Extraction en cours..."):
            extracted_data = extract_invoice_data(base64_img)

            st.markdown("### 📄 Données extraites")
            st.write(f"👤 Payer : {extracted_data['payer']['name']} ({extracted_data['payer']['account']})")
            st.write(f"👤 Payee : {extracted_data['payee']['name']} ({extracted_data['payee']['account']})")
            st.write(f"📅 Date : {extracted_data['date']}")
            st.write(f"💬 Raison : {extracted_data['reason']}")
            st.write(f"💶 Montant en lettres : {extracted_data['amount_words']}")

            st.markdown("### ✅ Résultats de validation")
            for check in validate_invoice_fields(extracted_data):
                st.write(f"- {check}")

# Footer
st.markdown("---")
st.markdown("© 2025 Chatbot Bancaire - Tous droits réservés.")
