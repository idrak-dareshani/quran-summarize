import os
import json

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
