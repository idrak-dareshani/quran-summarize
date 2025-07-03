import os
import streamlit as st
from translate import TafsirTranslator

DATA_DIR = "data"
CACHE_DIR = "cache"

# Page configuration
st.set_page_config(
    page_title="Quran Summarizer", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for scrollable columns and beautiful Urdu font
st.markdown("""
<style>
    /* Remove extra space above title */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
    }
            
    /* Scrollable containers */
    .scrollable-container {
        height: 60vh;
        overflow-y: auto;
        padding: 5px;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        background-color: #fafafa;
        margin-bottom: 10px;
    }
    
    /* Beautiful Urdu font styling */
    .urdu-text {
        font-family: 'Noto Nastaliq Urdu', 'Jameel Noori Nastaleeq', 'Nafees Web Naskh', serif;
        font-size: 18px;
        line-height: 2.2;
        text-align: right;
        direction: rtl;
        color: #2c3e50;
        padding: 10px;
    }
    
    /* English/other language text styling */
    .translated-text {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 16px;
        line-height: 1.8;
        color: #34495e;
        padding: 10px;
    }
    
    /* Custom scrollbar */
    .scrollable-container::-webkit-scrollbar {
        width: 8px;
    }
    
    .scrollable-container::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 4px;
    }
    
    .scrollable-container::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 4px;
    }
    
    .scrollable-container::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
</style>
""", unsafe_allow_html=True)

# Session state for caching
if 'translator' not in st.session_state:
    st.session_state.translator = TafsirTranslator()

st.title("📖 Quran Lecture Summarizer")

# Language selection
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

# Check if data directory exists
if not os.path.exists(DATA_DIR):
    st.error(f"Data directory '{DATA_DIR}' not found. Please create it and add text files.")
    st.stop()

# Get text files
text_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".txt")]
if not text_files:
    st.warning(f"No text files found in '{DATA_DIR}' directory.")
    st.stop()

text_files = sorted(text_files)

# Selection interface
col1, col2 = st.columns([3, 2])

with col1:
    selected_file = st.selectbox(
        "📁 Select Surah (Lecture)", 
        text_files, 
        key="file",
        help="Choose a lecture file to view"
    )

with col2:
    selected_lang_display = st.selectbox(
        "🌐 Translation Language", 
        list(language_map.keys()), 
        key="lang"
    )

selected_lang_code = language_map[selected_lang_display]

# Load original text
if selected_file:
    data_path = os.path.join(DATA_DIR, selected_file)
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            urdu_text = f.read().strip()
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        st.stop()

    # Translation logic with better error handling
    lang_file_path = os.path.join(CACHE_DIR, selected_lang_code, selected_file)
    translated_text = ""
    
    # Check cache first
    if os.path.exists(lang_file_path):
        try:
            with open(lang_file_path, 'r', encoding='utf-8') as f:
                translated_text = f.read()
        except Exception as e:
            st.warning(f"Cache read error: {str(e)}")
    
    # Translate if not cached
    if not translated_text:
        try:
            with st.spinner(f"Translating to {selected_lang_display}..."):
                result = st.session_state.translator.translate_tafsir(
                    urdu_text, "ur", selected_lang_code
                )
                translated_text = result["translated_text"]
                
                # Save to cache
                lang_folder = os.path.join(CACHE_DIR, selected_lang_code)
                os.makedirs(lang_folder, exist_ok=True)
                try:
                    with open(lang_file_path, 'w', encoding='utf-8') as f:
                        f.write(translated_text)
                    st.success("Translation cached for future use!")
                except Exception as e:
                    st.warning(f"Could not cache translation: {str(e)}")
                    
        except Exception as e:
            st.error(f"Translation failed: {str(e)}")
            translated_text = ""

    # Display content
    if translated_text:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"🗣 {selected_lang_display}")
            st.markdown(f"""
                <div class="scrollable-container">
                    <div class="translated-text">{translated_text}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.subheader("🗣 اردو (Original)")
            st.markdown(f"""
                <div class="scrollable-container">
                    <div class="urdu-text">{urdu_text}</div>
                </div>
            """, unsafe_allow_html=True)
    else:
        # Show only original if translation failed
        st.subheader("🗣 اردو (Original)")
        st.markdown(f"""
            <div class="scrollable-container">
                <div class="urdu-text">{urdu_text}</div>
            </div>
        """, unsafe_allow_html=True)
        
        st.info("💡 Translation not available. Please try selecting a different language.")
