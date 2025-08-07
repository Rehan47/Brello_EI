#!/usr/bin/env python3
"""
Setup script for Brello EI 0
Created by Epic Systems | Engineered by Rehan Temkar

Automated setup and installation for Brello EI 0 emotional intelligence model.
"""

import subprocess
import sys
import os
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        return False
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def check_cuda_availability():
    """Check CUDA availability"""
    print("\n🔧 Checking CUDA availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            return True
        else:
            print("⚠️  CUDA not available - will use CPU (slower)")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed - will install during setup")
        return False

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def check_model_access():
    """Check if user has access to the model"""
    print("\n🔐 Checking model access...")
    
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        
        # Try to access the model info
        model_info = api.model_info("meta-llama/Meta-Llama-3.2-3B-Instruct")
        print("✅ Model access confirmed!")
        return True
    except Exception as e:
        print("❌ Model access not available")
        print("💡 Please visit https://huggingface.co/meta-llama/Meta-Llama-3.2-3B-Instruct")
        print("   and accept the model license")
        return False

def test_model_loading():
    """Test if the model can be loaded"""
    print("\n🧪 Testing model loading...")
    
    try:
        from brello_ei_0 import BrelloEI0
        
        # Try loading with 4-bit quantization
        model = BrelloEI0(
            model_path="meta-llama/Meta-Llama-3.2-3B-Instruct",
            load_in_4bit=True
        )
        
        # Test a simple response
        response = model.generate_response("Hello!")
        print("✅ Model loaded and tested successfully!")
        return True
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def create_example_script():
    """Create an example usage script"""
    print("\n📝 Creating example script...")
    
    example_code = '''#!/usr/bin/env python3
"""
Quick Start - Brello EI 0
Created by Epic Systems | Engineered by Rehan Temkar
"""

from brello_ei_0 import BrelloEI0

def main():
    print("🤖 Brello EI 0 - Quick Start")
    print("=" * 30)
    
    # Load the model
    model = BrelloEI0(
        model_path="meta-llama/Meta-Llama-3.2-3B-Instruct",
        load_in_4bit=True
    )
    
    # Test conversation
    messages = [
        "I'm feeling really stressed about my presentation tomorrow.",
        "I just got promoted at work!",
        "I'm feeling overwhelmed with my responsibilities."
    ]
    
    for message in messages:
        print(f"\\nYou: {message}")
        response = model.generate_response(message)
        print(f"Brello EI 0: {response}")
        print("-" * 40)

if __name__ == "__main__":
    main()
'''
    
    with open("quick_start.py", "w") as f:
        f.write(example_code)
    
    print("✅ Created quick_start.py")

def main():
    """Run the complete setup process"""
    print("🤖 Brello EI 0 - Setup")
    print("Created by Epic Systems | Engineered by Rehan Temkar")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Check CUDA
    cuda_available = check_cuda_availability()
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Setup failed at dependency installation")
        return
    
    # Check model access
    if not check_model_access():
        print("\n⚠️  Please accept the model license and run setup again")
        return
    
    # Test model loading
    if not test_model_loading():
        print("\n❌ Setup failed at model loading test")
        return
    
    # Create example script
    create_example_script()
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Run: python quick_start.py")
    print("2. Run: python example_usage.py")
    print("3. Run: python test_brello_ei_0.py")
    print("4. Run: python train_brello_ei_0.py (to fine-tune)")
    
    if not cuda_available:
        print("\n⚠️  Note: Running on CPU will be slower. Consider using a GPU for better performance.")

if __name__ == "__main__":
    main()
