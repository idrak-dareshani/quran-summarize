import os
import json

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_summary(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

def chunk_text(text, max_words=500):
    words = text.split()
    for i in range(0, len(words), max_words):
        yield " ".join(words[i:i + max_words])

CACHE_DIR = "data/cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def load_cached_translation(file_key, lang_code):
    cache_file = os.path.join(CACHE_DIR, f"{file_key}_{lang_code}.json")
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f).get("text", None)
    return None

def save_translation_to_cache(file_key, lang_code, text):
    cache_file = os.path.join(CACHE_DIR, f"{file_key}_{lang_code}.json")
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump({"text": text}, f, ensure_ascii=False, indent=2)
