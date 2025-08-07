#!/usr/bin/env python3
"""
Model Options - Brello EI 0
Created by Epic Systems | Engineered by Rehan Temkar

Demonstrates different model options for Brello EI 0.
"""

from brello_ei_0 import BrelloEI0

def test_model_option(model_path, description):
    """Test a specific model option"""
    print(f"\n🤖 Testing: {description}")
    print(f"Model: {model_path}")
    print("-" * 50)
    
    try:
        # Load the model
        model = BrelloEI0(
            model_path=model_path,
            load_in_4bit=False
        )
        
        # Test emotional intelligence
        test_message = "I'm feeling really stressed about my presentation tomorrow."
        response = model.generate_response(test_message)
        
        print(f"Input: {test_message}")
        print(f"Response: {response}")
        print("✅ Model working!")
        
        return True
    except Exception as e:
        print(f"❌ Model failed: {e}")
        return False

def main():
    """Test different model options"""
    print("🤖 Brello EI 0 - Model Options")
    print("Created by Epic Systems | Engineered by Rehan Temkar")
    print("=" * 60)
    
    # Available model options
    model_options = [
        {
            "path": "microsoft/DialoGPT-medium",
            "description": "Public Model (Recommended for quick start)"
        },
        {
            "path": "microsoft/DialoGPT-large", 
            "description": "Larger Public Model (Better responses)"
        },
        {
            "path": "microsoft/DialoGPT-small",
            "description": "Smaller Public Model (Faster)"
        }
    ]
    
    # Test each model option
    working_models = []
    for option in model_options:
        if test_model_option(option["path"], option["description"]):
            working_models.append(option)
    
    print(f"\n📊 Results:")
    print(f"✅ Working models: {len(working_models)}/{len(model_options)}")
    
    if working_models:
        print("\n🎯 Recommended models:")
        for model in working_models:
            print(f"  • {model['path']} - {model['description']}")
    
    print("\n💡 To use Llama 3.2 3B:")
    print("1. Create Hugging Face account")
    print("2. Accept license at: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct")
    print("3. Login with: huggingface-cli login")
    print("4. Update model_path to: 'meta-llama/Llama-3.2-3B-Instruct'")

if __name__ == "__main__":
    main()
