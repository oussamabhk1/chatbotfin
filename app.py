# -*- coding: utf-8 -*-
import os
import json
import base64
import pandas as pd
from datetime import datetime
from PIL import Image
import io

import streamlit as st
from groq import Groq
from langdetect import detect
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

# Configuration de la page Streamlit
st.set_page_config(page_title="Chatbot Bancaire + Extraction Virements", layout="wide")
st.title("💬 Chatbot Bancaire")

# === INITIALISATION CHATBOT ===
@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

@st.cache_data
def load_data():
    return pd.read_csv("cleanedTranslatedBankFAQs.csv", usecols=[
        "Question", "Answer", "Class", "Profile",
        "Profile_fr", "Profile_ar", "Class_fr", "Class_ar",
        "Question_fr", "Question_ar", "Answer_fr", "Answer_ar"
    ])

model = load_model()
df = load_data()

@st.cache_resource
def build_embeddings(df):
    embeddings = {
        "fr": model.encode(df["Profile_fr"].fillna('') + " - " + df["Question_fr"].fillna('')),
        "en": model.encode(df["Profile"].fillna('') + " - " + df["Question"].fillna('')),
        "ar": model.encode(df["Profile_ar"].fillna('') + " - " + df["Question_ar"].fillna(''))
    }
    nn_models = {
        lang: NearestNeighbors(n_neighbors=1, metric="cosine").fit(embeddings[lang])
        for lang in embeddings
    }
    return embeddings, nn_models

embeddings, nn_models = build_embeddings(df)

# === INITIALISATION EXTRACTION virement ===
client = Groq(api_key="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")

def encode_image_file(uploaded_file):
    return base64.b64encode(uploaded_file.read()).decode("utf-8")

def extract_invoice_data(base64_image):
    system_prompt = """
    Extract payement data and return JSON with these exact fields:
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
    except:
        return False

def validate_invoice_fields(data):
    results = []
    results.append("✅ Payer name" if data['payer']['name'] else "❌ Missing payer name")
    results.append("✅ Payee name" if data['payee']['name'] else "❌ Missing payee name")
    results.append("✅ Payer account" if data['payer']['account'] and len(data['payer']['account']) == 8 else "❌ Invalid payer account")
    results.append("✅ Payee account" if data['payee']['account'] and len(data['payee']['account']) == 20 else "❌ Invalid payee account")
    results.append("✅ Valid date" if validate_date(data['date']) else "❌ Invalid or missing date")
    results.append("✅ reason" if data['reason'] else "❌ Missing reason")
    return results
# === INTERFACE STREAMLIT ===
tab1, tab2 = st.tabs(["📩 Chatbot Bancaire", "📤 Extraction Virements"])

with tab1:
    st.subheader("💬 Assistant Bancaire Intelligent")

    now = datetime.now().hour
    if now < 12:
        greeting = "☀️ Bonjour !"
    elif now < 18:
        greeting = "🌤️ Bon après-midi !"
    else:
        greeting = "🌙 Bonsoir !"

    st.markdown(f"### {greeting} Comment puis-je vous aider aujourd’hui ?")

    user_input = st.text_input("💡 Posez une question bancaire ci-dessous :")

    if user_input:
        lang = detect(user_input)
        query = model.encode(user_input)
        distances, indices = nn_models[lang].kneighbors([query])
        idx = indices[0][0]
        profile_col = f"Profile_{lang}" if lang != "en" else "Profile"
        profile_txt = df.iloc[idx][profile_col]
        # Afficher le profil
        st.write("### 🏷️ Profil concerné :")
        st.info(profile_txt)
        
        answer_col = f"Answer_{lang}" if lang != "en" else "Answer"
        st.write("### 📌 Réponse suggérée :")
        st.success(df.iloc[idx][answer_col])
with tab2:
    st.subheader("Uploader un virement à analyser")
    uploaded_file = st.file_uploader("📎 Déposez une image (.png/.jpg)", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        base64_img = encode_image_file(uploaded_file)
        st.image(uploaded_file, caption="Virement uploadée", use_column_width=True)
        with st.spinner("🧠 Extraction en cours..."):
            extracted_data = extract_invoice_data(base64_img)

            # Affichage personnalisé sans amount
            st.markdown("### 📄 Données extraites")
            st.write(f"👤 Payer : {extracted_data['payer']['name']} ({extracted_data['payer']['account']})")
            st.write(f"👤 Payee : {extracted_data['payee']['name']} ({extracted_data['payee']['account']})")
            st.write(f"📅 Date : {extracted_data['date']}")
            st.write(f"💬 Raison : {extracted_data['reason']}")
            st.write(f"💶 Montant en lettres : {extracted_data['amount_words']}")

            st.markdown("### ✅ Résultats de validation")
            for check in validate_invoice_fields(extracted_data):
                st.write(f"- {check}")

st.image("https://cdn-icons-png.flaticon.com/512/4712/4712109.png", width=60)
