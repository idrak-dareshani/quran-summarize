"""
Test script to verify all dependencies are installed correctly
"""

def test_imports():
    """Test if all required packages can be imported."""
    
    test_results = {}
    
    # Test core packages
    try:
        import torch
        test_results['torch'] = f"✅ PyTorch {torch.__version__}"
    except ImportError as e:
        test_results['torch'] = f"❌ PyTorch: {e}"
    
    try:
        import whisper
        test_results['whisper'] = "✅ OpenAI Whisper"
    except ImportError as e:
        test_results['whisper'] = f"❌ Whisper: {e}"
    
    try:
        import transformers
        test_results['transformers'] = f"✅ Transformers {transformers.__version__}"
    except ImportError as e:
        test_results['transformers'] = f"❌ Transformers: {e}"
    
    # Test optional packages
    try:
        import librosa
        test_results['librosa'] = f"✅ Librosa {librosa.__version__}"
    except ImportError as e:
        test_results['librosa'] = f"⚠️ Librosa (optional): {e}"
    
    try:
        import soundfile
        test_results['soundfile'] = f"✅ SoundFile {soundfile.__version__}"
    except ImportError as e:
        test_results['soundfile'] = f"⚠️ SoundFile (optional): {e}"
    
    return test_results

def test_whisper_models():
    """Test if Whisper models can be loaded."""
    try:
        import whisper
        
        print("📋 Available Whisper models:")
        for model_name in whisper.available_models():
            print(f"   - {model_name}")
        
        # Test loading a small model
        print("\n🔄 Testing model loading (this may take a moment)...")
        model = whisper.load_model("tiny")
        print("✅ Whisper model loaded successfully!")
        
        return True
    except Exception as e:
        print(f"❌ Error loading Whisper model: {e}")
        return False

def test_gpu_availability():
    """Check if GPU is available for processing."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"🚀 GPU Available: {gpu_name} (Count: {gpu_count})")
            return True
        else:
            print("💻 Using CPU (GPU not available)")
            return False
    except Exception as e:
        print(f"❌ Error checking GPU: {e}")
        return False

def main():
    """Run all tests."""
    print("🔍 Testing Audio Processing Pipeline Installation\n")
    
    # Test imports
    print("1️⃣ Testing Package Imports:")
    results = test_imports()
    for package, status in results.items():
        print(f"   {status}")
    
    print("\n" + "="*50 + "\n")
    
    # Test GPU
    print("2️⃣ Testing GPU Availability:")
    test_gpu_availability()
    
    print("\n" + "="*50 + "\n")
    
    # Test Whisper
    print("3️⃣ Testing Whisper Models:")
    whisper_ok = test_whisper_models()
    
    print("\n" + "="*50 + "\n")
    
    # Summary
    failed_imports = [pkg for pkg, status in results.items() if "❌" in status]
    
    if not failed_imports and whisper_ok:
        print("🎉 All tests passed! Your installation is ready.")
        print("\n📝 Next steps:")
        print("   1. Place your audio file in the 'audio/' directory")
        print("   2. Run: python main.py")
    else:
        print("⚠️ Some issues found:")
        if failed_imports:
            print(f"   - Failed imports: {', '.join(failed_imports)}")
        if not whisper_ok:
            print("   - Whisper model loading failed")
        print("\n🔧 Please install missing dependencies and try again.")

if __name__ == "__main__":
    main()