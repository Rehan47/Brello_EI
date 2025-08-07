#!/usr/bin/env python3
"""
Example Usage - Brello EI 0
Created by Epic Systems | Engineered by Rehan Temkar

Demonstrates how to use Brello EI 0 for emotionally intelligent conversations.
"""

from brello_ei_0 import BrelloEI0, load_brello_ei_0

def main():
    """Example usage of Brello EI 0"""
    
    print("🤖 Brello EI 0 - Emotional Intelligence AI")
    print("Created by Epic Systems | Engineered by Rehan Temkar")
    print("=" * 50)
    
    # Initialize the model
    print("📥 Loading Brello EI 0 model...")
    
    try:
        # Load the model with standard loading
        model = BrelloEI0(
            model_path="microsoft/DialoGPT-medium",
            load_in_4bit=False
        )
        
        print("✅ Model loaded successfully!")
        
        # Example conversations demonstrating emotional intelligence
        conversations = [
            "I'm feeling really anxious about my presentation tomorrow.",
            "I just got some great news and I'm so excited!",
            "I'm feeling overwhelmed with all my work lately.",
            "I'm really grateful for my friends and family.",
            "I'm not sure what I want to do with my career."
        ]
        
        print("\n💬 Example Emotional Intelligence Responses:")
        print("-" * 50)
        
        for i, message in enumerate(conversations, 1):
            print(f"\n{i}. User: {message}")
            response = model.generate_response(message)
            print(f"Brello EI 0: {response}")
            print("-" * 30)
        
        # Interactive chat
        print("\n💭 Interactive Chat Mode")
        print("Type 'quit' to exit")
        print("-" * 30)
        
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Brello EI 0: It's been wonderful talking with you! Take care and remember that your feelings matter. 💙")
                break
            
            if user_input:
                response = model.chat(user_input)
                print(f"Brello EI 0: {response}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("💡 Make sure you have the required dependencies installed:")
        print("pip install -r requirements.txt")
        print("\n💡 You may need to accept the model license on Hugging Face:")
        print("Visit: https://huggingface.co/meta-llama/Meta-Llama-3.2-3B-Instruct")

if __name__ == "__main__":
    main()
