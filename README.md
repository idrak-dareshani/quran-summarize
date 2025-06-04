# Transcribe

**Transcribe** is a Python-based project designed to transcribe audio files (`.mp3`, `.wav`) and summarize their content for improved comprehension. It is useful for converting speech to text and generating concise summaries, ideal for lectures, interviews, podcasts, and more.

## 📌 Description

This tool leverages state-of-the-art speech recognition and natural language processing (NLP) models to:
- Convert spoken content into written text using OpenAI’s Whisper.
- Clean and split the transcription into manageable sections.
- Generate summaries using transformer-based models for quick understanding.

## 🧰 Technologies Used

- **Python 3.13**
- **Jupyter Notebook**
- **OpenAI Whisper** – for transcription
- **NLTK** – for text preprocessing
- **Hugging Face Transformers** – for summarization

## 🛠 Installation

To set up the project locally:

# 1. Clone the repository
```bash
git clone https://github.com/idrak-dareshani/transcribe.git
cd transcribe
```
# 2. Install required packages
```bash
pip install git+https://github.com/openai/whisper.git
pip install nltk transformers time
```

> 💡 You can also open this project in **VS Code** or **PyCharm** and ensure your environment is set up with the above dependencies.

## 🚀 Usage

Run the Jupyter notebooks in the following sequence:

1. `transcribe.ipynb` – Load your audio file and transcribe the spoken content.
2. `cleanandsplit.ipynb` – Process and split the raw transcript.
3. `summarize.ipynb` – Generate summaries using NLP models.

> Note: For better performance and faster processing, you may use a GPU-enabled environment.

## 💡 Features

* Supports `.mp3` and `.wav` formats.
* Modular notebook structure for flexibility.
* Can be extended to support multilingual transcription and summarization.
* Ideal for educational, journalistic, and research purposes.

## 🤝 Contributing

Contributions are welcome! You may consider:

* Integrating larger Whisper models or GPU support.
* Adding multilingual support for transcription.
* Enhancing formatting and coherence of summaries.

To contribute, fork the repository, create a new branch, and submit a pull request.

## 📝 License

This project is licensed under the **GNU General Public License (GPL)**.

---

For issues, suggestions, or improvements, feel free to open a GitHub [Issue](https://github.com/idrak-dareshani/transcribe/issues).
