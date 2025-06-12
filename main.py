"""
Enhanced Audio Transcription and Summarization Pipeline for Urdu and English

This module provides functionality to:
1. Transcribe Urdu and English audio files using OpenAI Whisper
2. Process transcriptions into sentences using language-specific approaches
3. Generate summaries using appropriate multilingual models
4. Auto-detect language or use specified language
"""

import os
import re
import time
import logging
from pathlib import Path
from typing import List, Generator, Optional, Dict, Tuple

import whisper
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM


class AudioProcessor:
    """Handles multilingual audio transcription and text processing pipeline."""
    
    def __init__(self, model_size: str = "medium"):
        """
        Initialize the MultilingualAudioProcessor.
        
        Args:
            model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
        """
        self.model_size = model_size
        self.logger = self._setup_logger()
        
        # Initialize models
        self.whisper_model = None
        self.summarizer = None
        
        # Create necessary directories
        self._create_directories()
        
        # Language-specific patterns
        self.sentence_patterns = {
            'ur': r'[۔؟!]',  # Urdu sentence endings
            'en': r'[.!?]',   # English sentence endings
        }
        
        # Stop words for key phrase extraction
        self.stop_words = {
            'ur': {
                'اور', 'کا', 'کے', 'کی', 'میں', 'سے', 'کو', 'نے', 'ہے', 'ہیں', 'تھا', 'تھے', 
                'یہ', 'وہ', 'جو', 'کہ', 'لیے', 'ساتھ', 'بھی', 'تو', 'پر', 'اس', 'ان', 'ایک'
            },
            'en': {
                'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
                'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had',
                'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
                'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
            }
        }
        
    def _setup_logger(self) -> logging.Logger:
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('audio_processing.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def _create_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        directories = ['audio', 'transcripts', 'analysis']
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
            
    def _load_whisper_model(self) -> whisper.Whisper:
        """Load Whisper model with caching."""
        if self.whisper_model is None:
            self.logger.info(f"Loading Whisper model: {self.model_size}")
            self.whisper_model = whisper.load_model(self.model_size)
        return self.whisper_model
    
    def _load_summarizer(self, language: str = "ur"):
        """
        Load appropriate summarization model based on language.
        
        Args:
            language: Target language ('ur' for Urdu, 'en' for English)
        """
        if self.summarizer is None:
            self.logger.info(f"Loading summarization model for language: {language}")
            try:
                if language == "en":
                    # Use BART for English - better performance
                    model_name = "facebook/bart-large-cnn"
                    self.logger.info("Loading BART model for English")
                else:
                    # Use mT5 for Urdu and other languages
                    model_name = "google/mt5-small"
                    self.logger.info("Loading mT5 model for Urdu/multilingual")
                
                self.summarizer = pipeline(
                    "summarization", 
                    model=model_name,
                    tokenizer=model_name,
                    device=0 if self._is_gpu_available() else -1
                )
            except Exception as e:
                self.logger.warning(f"Failed to load preferred model: {e}")
                # Fallback to BART
                self.logger.info("Falling back to BART model")
                self.summarizer = pipeline(
                    "summarization", 
                    model="facebook/bart-large-cnn",
                    device=0 if self._is_gpu_available() else -1
                )
        return self.summarizer
    
    def _is_gpu_available(self) -> bool:
        """Check if GPU is available for processing."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def detect_language(self, text: str) -> str:
        """
        Simple language detection based on character sets.
        
        Args:
            text: Text to analyze
            
        Returns:
            Language code ('ur' or 'en')
        """
        # Count Urdu/Arabic characters
        urdu_chars = len(re.findall(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]', text))
        
        # Count English characters
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        
        # Determine primary language
        if urdu_chars > english_chars:
            return 'ur'
        else:
            return 'en'
    
    def transcribe_audio(self, audio_file: str, language: str = "ur", 
                        output_file: str = None, auto_detect: bool = False) -> Tuple[str, str]:
        """
        Transcribe audio file to text with multilingual support.
        
        Args:
            audio_file: Path to audio file
            language: Language code for transcription (default: 'ur' for Urdu)
            output_file: Path to save transcription (auto-generated if None)
            auto_detect: Whether to auto-detect language after initial transcription
            
        Returns:
            Tuple of (transcribed_text, detected_language)
        """
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        
        model = self._load_whisper_model()
        
        # Generate output filename if not provided
        if output_file is None:
            base_name = Path(audio_file).stem
            output_file = f"transcripts/{base_name}_transcription.txt"
        
        self.logger.info(f"Starting transcription of {audio_file} (language: {language})")
        start_time = time.time()
        
        try:
            # Enhanced options for better transcription
            result = model.transcribe(
                audio_file, 
                language=language if not auto_detect else None,
                fp16=False,  # Better for CPU processing
                verbose=True,
                word_timestamps=True  # Get word-level timestamps
            )
            
            transcription = result["text"]
            
            # Auto-detect language if requested
            detected_language = language
            if auto_detect:
                detected_language = self.detect_language(transcription)
                self.logger.info(f"Auto-detected language: {detected_language}")
                
                # Re-transcribe with detected language if different
                if detected_language != language:
                    self.logger.info(f"Re-transcribing with detected language: {detected_language}")
                    result = model.transcribe(
                        audio_file, 
                        language=detected_language,
                        fp16=False,
                        verbose=True,
                        word_timestamps=True
                    )
                    transcription = result["text"]
            
            # Post-process transcription based on language
            transcription = self._clean_text(transcription, detected_language)
            
            # Calculate processing time
            elapsed = time.time() - start_time
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            self.logger.info(f"Transcription completed in {minutes}m {seconds}s")
            
            # Save transcription with UTF-8 encoding
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(f"Language: {detected_language}\n")
                f.write(f"Transcription:\n{transcription}")
            
            # Also save detailed results if available
            if "segments" in result:
                detailed_file = output_file.replace(".txt", "_detailed.txt")
                with open(detailed_file, "w", encoding="utf-8") as f:
                    f.write(f"Language: {detected_language}\n")
                    f.write("Detailed Transcription with Timestamps:\n\n")
                    for segment in result["segments"]:
                        f.write(f"[{segment['start']:.2f}s - {segment['end']:.2f}s]: {segment['text']}\n")
                self.logger.info(f"Detailed transcription saved to {detailed_file}")
            
            self.logger.info(f"Transcription saved to {output_file}")
            return transcription, detected_language
            
        except Exception as e:
            self.logger.error(f"Error during transcription: {str(e)}")
            raise
    
    def _clean_text(self, text: str, language: str) -> str:
        """
        Clean and format text based on language.
        
        Args:
            text: Raw transcribed text
            language: Language code
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        if language == 'ur':
            # Urdu-specific cleaning
            text = re.sub(r'\s+([۔؟!])', r'\1', text)  # Remove space before punctuation
            text = re.sub(r'([۔؟!])\s*', r'\1 ', text)  # Ensure space after punctuation
            # Keep Urdu, Arabic, and basic punctuation
            text = re.sub(r'[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF\s۔؟!،٫٬\w]', '', text)
        else:
            # English-specific cleaning
            text = re.sub(r'\s+([.!?])', r'\1', text)  # Remove space before punctuation
            text = re.sub(r'([.!?])\s*', r'\1 ', text)  # Ensure space after punctuation
            # Basic punctuation cleanup for English
            text = re.sub(r'[^\w\s.!?,:;\'\"()-]', '', text)
        
        return text.strip()
    
    def process_sentences_regex(self, text: str, language: str, 
                               output_file: str = None) -> List[str]:
        """
        Process text into sentences using language-specific regex patterns.
        
        Args:
            text: Input text to process
            language: Language code
            output_file: Path to save sentences (auto-generated if None)
            
        Returns:
            List of sentences
        """
        if output_file is None:
            output_file = f"analysis/sentences_{language}.txt"
        
        # Get language-specific pattern
        pattern = self.sentence_patterns.get(language, self.sentence_patterns['en'])
        
        # Split on sentence endings
        sentences = re.split(pattern, text)
        
        # Clean and filter sentences
        processed_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence.split()) >= 3:  # Minimum 3 words
                processed_sentences.append(sentence)
        
        # Add back sentence endings (except for the last one)
        final_sentences = []
        ending_char = '۔' if language == 'ur' else '.'
        
        for i, sentence in enumerate(processed_sentences[:-1]):
            # Find the original ending punctuation
            next_pos = text.find(sentence) + len(sentence)
            if next_pos < len(text) and text[next_pos] in ('۔؟!' if language == 'ur' else '.!?'):
                ending = text[next_pos]
                final_sentences.append(sentence + ending)
            else:
                final_sentences.append(sentence + ending_char)
        
        # Add the last sentence without ending if it exists
        if processed_sentences:
            final_sentences.append(processed_sentences[-1])
        
        # Save sentences
        with open(output_file, 'w', encoding="utf-8") as f:
            f.write(f"Language: {language}\n")
            f.write(f"Total sentences: {len(final_sentences)}\n\n")
            for i, sentence in enumerate(final_sentences, 1):
                f.write(f"{i}. {sentence.strip()}\n")
        
        self.logger.info(f"Processed {len(final_sentences)} sentences for {language}, saved to {output_file}")
        return final_sentences
    
    def chunk_text(self, text: str, language: str, max_words: int = None) -> Generator[str, None, None]:
        """
        Split text into chunks for processing based on language.
        
        Args:
            text: Text to chunk
            language: Language code
            max_words: Maximum words per chunk (auto-adjusted by language if None)
            
        Yields:
            Text chunks
        """
        if max_words is None:
            # Adjust chunk size based on language
            max_words = 300 if language == 'ur' else 500
        
        words = text.split()
        for i in range(0, len(words), max_words):
            chunk = ' '.join(words[i:i + max_words])
            if chunk.strip():
                yield chunk
    
    def summarize_text(self, text: str, language: str, max_length: int = None, 
                      min_length: int = None) -> str:
        """
        Summarize text using appropriate models based on language.
        
        Args:
            text: Text to summarize
            language: Language code
            max_length: Maximum summary length (auto-adjusted if None)
            min_length: Minimum summary length (auto-adjusted if None)
            
        Returns:
            Summarized text
        """
        if not text.strip():
            return "⚠️ Empty input. Skipped."
        
        word_count = len(text.split())
        if word_count < 20:
            return "ℹ️ Text too short to summarize. Consider keeping as-is."
        
        # Adjust parameters based on language
        if max_length is None:
            max_length = 17 if language == 'ur' else 150
        if min_length is None:
            min_length = 8 if language == 'ur' else 50
        
        summarizer = self._load_summarizer(language)
        summaries = []
        
        self.logger.info(f"Summarizing {language} text with {word_count} words")
        
        try:
            for i, chunk in enumerate(self.chunk_text(text, language), 1):
                try:
                    self.logger.debug(f"Processing chunk {i}")
                    
                    # For mT5, we might need to add a prefix
                    if (hasattr(summarizer.model, 'config') and 
                        'mt5' in str(summarizer.model.config).lower()):
                        summary_input = f"summarize: {chunk}"
                    else:
                        summary_input = chunk
                    
                    summary = summarizer(
                        summary_input, 
                        max_length=max_length, 
                        min_length=min_length, 
                        do_sample=False,
                        truncation=True
                    )
                    summaries.append(summary[0]['summary_text'])
                    
                except Exception as e:
                    error_msg = f"❌ Error summarizing chunk {i}: {str(e)}"
                    self.logger.error(error_msg)
                    # Fallback: return first few sentences
                    if language == 'ur':
                        sentences = chunk.split('۔')[:3]
                        fallback_summary = '۔'.join(sentences[:2]) + '۔'
                    else:
                        sentences = chunk.split('.')[:3]
                        fallback_summary = '.'.join(sentences[:2]) + '.'
                    summaries.append(f"📝 {fallback_summary}")
            
            return "\n\n".join(summaries)
            
        except Exception as e:
            self.logger.error(f"Summarization failed: {str(e)}")
            # Return first few sentences as fallback
            if language == 'ur':
                sentences = text.split('۔')[:3]
                return '۔'.join(sentences[:2]) + '۔'
            else:
                sentences = text.split('.')[:3]
                return '.'.join(sentences[:2]) + '.'
    
    def extract_key_phrases(self, text: str, language: str) -> List[str]:
        """
        Extract key phrases from text using language-specific frequency analysis.
        
        Args:
            text: Input text
            language: Language code
            
        Returns:
            List of key phrases
        """
        if language == 'ur':
            # Extract Urdu words
            words = re.findall(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]+', text)
        else:
            # Extract English words
            words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        
        # Get language-specific stop words
        stop_words = self.stop_words.get(language, self.stop_words['en'])
        
        # Count word frequency
        word_freq = {}
        for word in words:
            if word not in stop_words and len(word) > 2:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top words
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return [word for word, freq in top_words]
    
    def process_pipeline(self, file_name: str, language: str = "ur", 
                        auto_detect: bool = False) -> dict:
        """
        Run the complete multilingual audio processing pipeline.
        
        Args:
            file_name: Name of the audio file (without extension)
            language: Language code for transcription (default: 'ur')
            auto_detect: Whether to auto-detect language
            
        Returns:
            Dictionary with processing results
        """
        results = {}
        
        try:
            # Step 1: Transcribe audio
            audio_file = f"audio/{file_name}.mp3"
            transcription, detected_language = self.transcribe_audio(
                audio_file, language, auto_detect=auto_detect
            )
            results['transcription'] = transcription
            results['language'] = detected_language
            results['word_count'] = len(transcription.split())
            
            # Step 2: Process sentences using regex
            sentences = self.process_sentences_regex(transcription, detected_language)
            results['sentences'] = sentences
            results['sentence_count'] = len(sentences)
            
            # Step 3: Extract key phrases
            key_phrases = self.extract_key_phrases(transcription, detected_language)
            results['key_phrases'] = key_phrases
            
            # Step 4: Generate summary
            combined_text = " ".join(sentences)
            summary = self.summarize_text(combined_text, detected_language)
            
            # Save summary
            summary_file = f"analysis/{file_name}_{detected_language}_summary.txt"
            with open(summary_file, "w", encoding="utf-8") as f:
                if detected_language == 'ur':
                    f.write("=== خلاصہ (Summary) ===\n")
                    f.write(summary)
                    f.write("\n\n=== اہم الفاظ (Key Phrases) ===\n")
                else:
                    f.write("=== Summary ===\n")
                    f.write(summary)
                    f.write("\n\n=== Key Phrases ===\n")
                f.write(", ".join(key_phrases))
            
            results['summary'] = summary
            results['summary_file'] = summary_file
            results['status'] = 'success'
            
            self.logger.info(f"Multilingual audio processing pipeline completed successfully for {detected_language}")
            
        except Exception as e:
            error_msg = f"Pipeline failed: {str(e)}"
            self.logger.error(error_msg)
            results['status'] = 'error'
            results['error'] = error_msg
            
        return results


def main():
    """Main function to run the multilingual audio processing pipeline."""
    # Configuration
    file_name = "001 - SURAH AL-FATIAH"  # Your audio file name (without extension)
    language = "ur"  # Default to Urdu, but can be "en" for English
    model_size = "medium"  # Recommended for better accuracy
    auto_detect = False  # Set to True to auto-detect language
    
    # Initialize processor
    processor = AudioProcessor(model_size=model_size)
    
    # Run pipeline
    results = processor.process_pipeline(file_name, language, auto_detect=auto_detect)
    
    # Print results
    if results['status'] == 'success':
        lang_name = "Urdu" if results['language'] == 'ur' else "English"
        print(f"✅ {lang_name} audio processing completed successfully!")
        print(f"🗣️ Detected language: {lang_name}")
        print(f"📊 Word count: {results['word_count']}")
        print(f"📝 Processed {results['sentence_count']} sentences")
        print(f"🔑 Key phrases: {', '.join(results['key_phrases'][:5])}")
        print(f"📄 Summary saved to {results['summary_file']}")
        
        # Display first few lines of summary
        print(f"\n📋 Summary preview:")
        summary_preview = results['summary'][:200] + "..." if len(results['summary']) > 200 else results['summary']
        print(summary_preview)
        
    else:
        print(f"❌ Processing failed: {results['error']}")


if __name__ == "__main__":
    main()