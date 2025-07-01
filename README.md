# 🎙️ Quran Summarize – Multilingual Audio & Text Summarization

**Quran Summarize** is a comprehensive pipeline for transcribing and summarizing Urdu and English audio files, as well as summarizing Quranic text files. Built with Whisper, Transformer models, and a user-friendly Streamlit web interface, it automates the process from raw `.mp3` or `.txt` input to structured summaries and key phrase extraction.

## 📁 Project Structure

```
.
├── app.py                # Streamlit web app entry point
├── requirements.txt
├── README.md
├── data/                 # Input text files for summarization
├── src/
│   └── summarize_all.py  # Batch summarization script for text files
└── utils/
```

## 🛠 Technologies Used

- [Python](https://www.python.org/)
- [Streamlit](https://streamlit.io/)
- [Whisper](https://github.com/openai/whisper)
- [Transformers (HuggingFace)](https://huggingface.co/docs/transformers/)
- [Torch](https://pytorch.org/)
- [Regex, Logging, Unicode normalization]

## 🧰 Installation

```bash
# 1. Clone the repository
git clone https://github.com/idrak-dareshani/quran-summarize.git
cd quran-summarize
```

```bash
# 2. Set up a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

```bash
# 3. Install required packages
pip install -r requirements.txt
```
⚠️ Ensure your system has ffmpeg installed. Whisper requires it to process audio.

## 🚀 Usage

### Web Interface (Recommended)

Launch the Streamlit app:

```bash
streamlit run app.py
```

#### Text File Summarization

To summarize all text files in the `data/` directory, run:

```bash
python src/summarize_all.py
```

## 🤝 Contributing

* Contributions are welcome!
* Improve model selection or chunking strategies
* Add support for more languages
* Enhance sentence segmentation or output formatting
* Add GUI or Web interface features

Please fork the repository, make your changes, and submit a pull request.

## 📄 License

This project is licensed under the GNU General Public License (GPL).

For questions or support, feel free to open an [issue.](https://github.com/idrak-dareshani/quran-summarize/issues)