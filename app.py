import streamlit as st
import os
import json
from deep_translator import GoogleTranslator
from src.utils import load_cached_translation, save_translation_to_cache

JSON_DIR = "data/json"

st.set_page_config(layout="wide")
st.title("📖 Quranic Lecture Summarizer")

# Language selection (excluding English)
language_map = {
    "اردو (Urdu)": "ur",
    "Arabic (العربية)": "ar",
    "Turkish (Türkçe)": "tr",
    "Hindi (हिन्दी)": "hi",
    "French (Français)": "fr",
    "Bengali (বাংলা)": "bn",
    "German (Deutsch)": "de"
}

# 🔄 Dropdowns on the same row
col1, col2 = st.columns([2, 1])

with col1:
    json_files = sorted(f for f in os.listdir(JSON_DIR) if f.endswith(".json"))
    selected_file = st.selectbox("📁 Select Surah (Lecture)", json_files, key="file")

with col2:
    selected_lang_display = st.selectbox("🌐 View alongside English", list(language_map.keys()), key="lang")
    selected_lang_code = language_map[selected_lang_display]

# 🧾 Display logic
if selected_file:
    json_path = os.path.join(JSON_DIR, selected_file)

    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            urdu_text = json_data.get("corrected_text", "❌ No corrected text found.")
    else:
        urdu_text = "❌ JSON file not found."

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