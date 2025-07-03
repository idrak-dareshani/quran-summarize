import streamlit as st
import os
from translate import TafsirTranslator

DATA_DIR = "data"
CACHE_DIR = "cache"

st.set_page_config(page_title="Quran Summarizer", layout="wide")
st.title("📖 Quran Lecture Summarizer")

# Language selection (excluding English)
language_map = {
    "Arabic": "ar",
    "Bengali": "bn",
    "English": "en",
    "French": "fr",
    "German": "de",
    "Hindi": "hi",
    "Indonesian": "id",
    "Malay": "ms",
    "Spanish": "es",
    "Swahili": "sw",
    "Turkish": "tr"
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

    # 🌐 Load translated file from cache
    try:
        
        # en_file_path = os.path.join(CACHE_DIR, "en", selected_file)
        # if os.path.exists(en_file_path):
        #     with open(en_file_path, 'r', encoding='utf-8') as f:
        #         translated_english = f.read()

        lang_file_path = os.path.join(CACHE_DIR, selected_lang_code, selected_file)
        if os.path.exists(lang_file_path):
            with open(lang_file_path, 'r', encoding='utf-8') as f:
                translated_selected = f.read()
        else:
            translator = TafsirTranslator()
            result = translator.translate_tafsir(urdu_text, "ur", selected_lang_code)
            translated_selected = result["translated_text"]
            
            # save in cache folder for next
            lang_folder = os.path.join(CACHE_DIR, selected_lang_code)
            os.makedirs(lang_folder, exist_ok=True)
            with open(lang_file_path, 'w', encoding='utf-8') as f:
                f.write(translated_selected)

    except Exception as e:
        st.error(f"❌ Translation failed: {str(e)}")
        translated_selected = ""
        #translated_english = ""

    # 📖 Display in 2 columns
    col1, col2 = st.columns(2)
    with col2:
        st.subheader("🗣 Urdu (Original)")
        st.write(urdu_text)

    with col1:
        st.subheader(f"🗣 {selected_lang_display}")
        st.write(translated_selected)