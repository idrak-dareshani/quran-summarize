from deep_translator import GoogleTranslator
from transformers import pipeline
from src.utils import load_json, save_summary, chunk_text

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_file(json_path, summary_path):
    data = load_json(json_path)
    urdu_text = data.get("corrected_text") or data.get("original_text", "")

    if len(urdu_text.split()) < 30:
        return "Too short to summarize."

    # Step 1: Translate Urdu ➝ English
    try:
        english_text = GoogleTranslator(source='ur', target='en').translate(urdu_text)
    except Exception as e:
        return f"Translation failed: {str(e)}"

    # Step 2: Summarize
    chunks = list(chunk_text(english_text))
    summaries = []

    for chunk in chunks:
        try:
            result = summarizer(chunk, max_length=80, min_length=30, do_sample=False)
            summaries.append(result[0]['summary_text'])
        except Exception as e:
            summaries.append(f"Error summarizing chunk: {str(e)}")

    summary_text_en = "\n\n".join(summaries)

    # Optional: Translate back to Urdu
    try:
        summary_text_ur = GoogleTranslator(source='en', target='ur').translate(summary_text_en)
    except Exception as e:
        summary_text_ur = "❌ Error translating summary back to Urdu: " + str(e)

    # Save summary in Urdu
    save_summary(summary_path, summary_text_ur)
    return summary_text_ur
