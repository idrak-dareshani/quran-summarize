import streamlit as st
import os
import json
from deep_translator import GoogleTranslator
from src.utils import load_cached_translation, save_translation_to_cache

DATA_DIR = "data"

st.set_page_config(page_title="Quran Summarizer", layout="wide")
st.title("📖 Quranic Lecture Summarizer")

# Language selection (excluding English)
language_map = {
    "اردو (Urdu)": "ur",
    "Turkish (Türkçe)": "tr",
    "Hindi (हिन्दी)": "hi",
    "French (Français)": "fr",
    "Bengali (বাংলা)": "bn",
    "German (Deutsch)": "de"
}

# 🔄 Dropdowns on the same row
col1, col2 = st.columns([2, 1])

with col1:
    text_files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".txt"))
    selected_file = st.selectbox("📁 Select Surah (Lecture)", text_files, key="file")

with col2:
    selected_lang_display = st.selectbox("🌐 View alongside English", list(language_map.keys()), key="lang")
    selected_lang_code = language_map[selected_lang_display]

# 🧾 Display logic
if selected_file:
    data_path = os.path.join(DATA_DIR, selected_file)

    if os.path.exists(data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            text_data = f.read()
            urdu_text = text_data.strip()
    else:
        urdu_text = "❌ TXT file not found."

    #st.markdown("---")
    #st.subheader(f"📜 Translation of {selected_file.replace('.json','')}")

    # Remove extension and make cache-safe key
    file_key = os.path.splitext(selected_file)[0].replace(" ", "_")

    # 🌐 Translation
    try:
        # Check cache first
        translated_selected = load_cached_translation(file_key, selected_lang_code)
        translated_english = load_cached_translation(file_key, "en")

        # Translate and save if not in cache
        if not translated_selected:
            translated_selected = urdu_text if selected_lang_code == "ur" else \
                GoogleTranslator(source='ur', target=selected_lang_code).translate(urdu_text)
            save_translation_to_cache(file_key, selected_lang_code, translated_selected)

        if not translated_english:
            translated_english = GoogleTranslator(source='ur', target='en').translate(urdu_text)
            save_translation_to_cache(file_key, "en", translated_english)

    except Exception as e:
        st.error(f"❌ Translation failed: {str(e)}")
        translated_selected = ""
        translated_english = ""

    # 📖 Display in 2 columns
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🗣 English")
        st.write(translated_english)

    with col2:
        st.subheader(f"🗣 {selected_lang_display}")
        st.write(translated_selected)