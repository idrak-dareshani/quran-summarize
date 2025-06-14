# 🎙️ Transcribe – Multilingual Audio Transcription & Summarization

**Transcribe** is a comprehensive pipeline for transcribing and summarizing Urdu and English audio files. Built with Whisper and Transformer models, it automates the process from raw `.mp3` input to structured summaries and key phrase extraction.

## 📌 Features

- 🎧 Transcribes audio files in **Urdu** and **English** using OpenAI's Whisper.
- 🧠 Detects language and loads appropriate **summarization model** (BART for English, mT5 for Urdu).
- ✂️ Cleans and splits transcription into structured sentences.
- 📑 Generates summaries and extracts key phrases.
- 📂 Supports both **single file** and **batch processing** modes.
- 💻 CPU and GPU support for summarization (automatically selected).

## 🛠 Technologies Used

- [Python](https://www.python.org/)
- [Whisper](https://github.com/openai/whisper)
- [Transformers (HuggingFace)](https://huggingface.co/docs/transformers/)
- [Torch](https://pytorch.org/)
- [Regex, Logging, Unicode normalization]

## 🧰 Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/transcribe.git
cd transcribe
```

```bash
# 2. Set up a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

```bash
# 3. Install required packages
pip install git+https://github.com/openai/whisper.git
pip install transformers torch
```
⚠️ Ensure your system has ffmpeg installed. Whisper requires it to process audio.

## 🚀 Usage
📁 Audio File Placement
Place your .mp3 files inside the audio/ directory. Use clear file names (e.g., 001 - SURAH AL-FATIAH.mp3).

## ▶️ Run Transcription & Summarization
### Single File Mode
Edit the file_name variable in main.py and run:
```bash
python main.py
```
### Batch Mode
Enable batch_mode = True in main.py to automatically process all .mp3 files in the audio/ directory.

## 📂 Output
* Transcriptions saved to transcripts/
* Sentence splits and summaries saved to analysis/
* Key phrases displayed and saved in summary files

🌐 Languages Supported
*Urdu (default)
*English

## 🧪 Example Output
```text
✅ Urdu audio processing completed successfully!
🗣️ Detected language: Urdu
📊 Word count: 587
📝 Processed 38 sentences
🔑 Key phrases: قرآن, اللہ, انسان, علم, زندگی
📄 Summary saved to analysis/001 - SURAH AL-FATIAH_ur_summary.txt
```

## 🤝 Contributing
* Contributions are welcome!
* Improve model selection or chunking strategies
* Add support for more languages
* Enhance sentence segmentation or output formatting
* Add GUI or Web interface

Please fork the repository, make your changes, and submit a pull request.

## 📄 License
This project is licensed under the GNU General Public License (GPL).

For questions or support, feel free to open an [issue.](https://github.com/idrak-dareshani/transcribe_audio_file/issues)
