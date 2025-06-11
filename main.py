"""
Audio Transcription and Summarization Pipeline

This module provides functionality to:
1. Transcribe audio files using OpenAI Whisper
2. Process transcriptions into sentences
3. Generate summaries using BART model
"""

import os
import time
import logging
from pathlib import Path
from typing import List, Generator, Optional

import whisper
import nltk
from nltk.tokenize import sent_tokenize
from transformers import pipeline


class AudioProcessor:
    """Handles audio transcription and text processing pipeline."""
    
    def __init__(self, model_size: str = "medium", nltk_data_path: str = "nltk_data"):
        """
        Initialize the AudioProcessor.
        
        Args:
            model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
            nltk_data_path: Path to store NLTK data
        """
        self.model_size = model_size
        self.nltk_data_path = nltk_data_path
        self.logger = self._setup_logger()
        
        # Initialize models
        self.whisper_model = None
        self.summarizer = None
        
        # Create necessary directories
        self._create_directories()
        
    def _setup_logger(self) -> logging.Logger:
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('audio_processing.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def _create_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        directories = ['audio', 'transcripts', 'analysis', self.nltk_data_path]
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
            
    def _load_whisper_model(self) -> whisper.Whisper:
        """Load Whisper model with caching."""
        if self.whisper_model is None:
            self.logger.info(f"Loading Whisper model: {self.model_size}")
            self.whisper_model = whisper.load_model(self.model_size)
        return self.whisper_model
    
    def _load_summarizer(self):
        """Load BART summarization model with caching."""
        if self.summarizer is None:
            self.logger.info("Loading BART summarization model")
            self.summarizer = pipeline(
                "summarization", 
                model="facebook/bart-large-cnn",
                device=0 if self._is_gpu_available() else -1  # Use GPU if available
            )
        return self.summarizer
    
    def _is_gpu_available(self) -> bool:
        """Check if GPU is available for processing."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def transcribe_audio(self, audio_file: str, language: str = "en", 
                        output_file: str = "transcripts/transcription.txt") -> str:
        """
        Transcribe audio file to text.
        
        Args:
            audio_file: Path to audio file
            language: Language code for transcription
            output_file: Path to save transcription
            
        Returns:
            Transcribed text
        """
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        
        model = self._load_whisper_model()
        
        self.logger.info(f"Starting transcription of {audio_file}")
        start_time = time.time()
        
        try:
            result = model.transcribe(audio_file, language=language)
            transcription = result["text"]
            
            # Calculate processing time
            elapsed = time.time() - start_time
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            self.logger.info(f"Transcription completed in {minutes}m {seconds}s")
            
            # Save transcription
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(transcription)
            
            self.logger.info(f"Transcription saved to {output_file}")
            return transcription
            
        except Exception as e:
            self.logger.error(f"Error during transcription: {str(e)}")
            raise
    
    def process_sentences(self, text: str, output_file: str = "analysis/sentences.txt") -> List[str]:
        """
        Process text into sentences using NLTK.
        
        Args:
            text: Input text to process
            output_file: Path to save sentences
            
        Returns:
            List of sentences
        """
        # Download NLTK data if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            self.logger.info("Downloading NLTK punkt tokenizer")
            nltk.download('punkt_tab', download_dir=self.nltk_data_path, quiet=True)
        
        # Tokenize sentences
        sentences = sent_tokenize(text, language='english')
        
        # Save sentences
        with open(output_file, 'w', encoding="utf-8") as f:
            for sentence in sentences:
                f.write(sentence.strip() + '\n')
        
        self.logger.info(f"Processed {len(sentences)} sentences, saved to {output_file}")
        return sentences
    
    def chunk_text(self, text: str, max_words: int = 500) -> Generator[str, None, None]:
        """
        Split text into chunks for processing.
        
        Args:
            text: Text to chunk
            max_words: Maximum words per chunk
            
        Yields:
            Text chunks
        """
        words = text.split()
        for i in range(0, len(words), max_words):
            yield ' '.join(words[i:i + max_words])
    
    def summarize_text(self, text: str, max_length: int = 80, 
                      min_length: int = 30) -> str:
        """
        Summarize text using BART model with chunking support.
        
        Args:
            text: Text to summarize
            max_length: Maximum summary length
            min_length: Minimum summary length
            
        Returns:
            Summarized text
        """
        if not text.strip():
            return "⚠️ Empty input. Skipped."
        
        word_count = len(text.split())
        if word_count < 30:
            return "ℹ️ Text too short to summarize. Consider keeping as-is."
        
        summarizer = self._load_summarizer()
        summaries = []
        
        self.logger.info(f"Summarizing text with {word_count} words")
        
        for i, chunk in enumerate(self.chunk_text(text), 1):
            try:
                self.logger.debug(f"Processing chunk {i}")
                summary = summarizer(
                    chunk, 
                    max_length=max_length, 
                    min_length=min_length, 
                    do_sample=False
                )
                summaries.append(summary[0]['summary_text'])
                
            except Exception as e:
                error_msg = f"❌ Error summarizing chunk {i}: {str(e)}"
                self.logger.error(error_msg)
                summaries.append(error_msg)
        
        return "\n\n".join(summaries)
    
    def process_pipeline(self, audio_file: str, language: str = "en") -> dict:
        """
        Run the complete audio processing pipeline.
        
        Args:
            audio_file: Path to audio file
            language: Language code for transcription
            
        Returns:
            Dictionary with processing results
        """
        results = {}
        
        try:
            # Step 1: Transcribe audio
            transcription = self.transcribe_audio(audio_file, language)
            results['transcription'] = transcription
            
            # Step 2: Process sentences
            sentences = self.process_sentences(transcription)
            results['sentences'] = sentences
            results['sentence_count'] = len(sentences)
            
            # Step 3: Generate summary
            combined_text = " ".join(sentences)
            summary = self.summarize_text(combined_text)
            
            # Save summary
            with open("analysis/summary.txt", "w", encoding="utf-8") as f:
                f.write(summary)
            
            results['summary'] = summary
            results['status'] = 'success'
            
            self.logger.info("Pipeline completed successfully")
            
        except Exception as e:
            error_msg = f"Pipeline failed: {str(e)}"
            self.logger.error(error_msg)
            results['status'] = 'error'
            results['error'] = error_msg
            
        return results


def main():
    """Main function to run the audio processing pipeline."""
    # Configuration
    audio_file = "audio/input_audio.mp3"
    language = "en"
    model_size = "medium"
    
    # Initialize processor
    processor = AudioProcessor(model_size=model_size)
    
    # Run pipeline
    results = processor.process_pipeline(audio_file, language)
    
    # Print results
    if results['status'] == 'success':
        print(f"✅ Processing completed successfully!")
        print(f"📊 Processed {results['sentence_count']} sentences")
        print(f"📝 Summary saved to analysis/summary.txt")
    else:
        print(f"❌ Processing failed: {results['error']}")


if __name__ == "__main__":
    main()