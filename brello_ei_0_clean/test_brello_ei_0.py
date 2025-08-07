#!/usr/bin/env python3
"""
Test Brello EI 0 - Emotional Intelligence Model
Created by Epic Systems | Engineered by Rehan Temkar

Test script to verify Brello EI 0 functionality and emotional intelligence capabilities.
"""

import torch
from brello_ei_0 import BrelloEI0
import time

def test_model_loading():
    """Test model loading functionality"""
    print("🧪 Testing Model Loading...")
    
    try:
        model = BrelloEI0(
            model_path="microsoft/DialoGPT-medium",
            load_in_4bit=False
        )
        print("✅ Model loaded successfully!")
        return model
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return None

def test_emotional_intelligence_responses(model):
    """Test emotional intelligence response generation"""
    print("\n🧪 Testing Emotional Intelligence Responses...")
    
    test_cases = [
        {
            "input": "I'm feeling really anxious about my job interview tomorrow.",
            "expected_keywords": ["understand", "anxious", "natural", "stress", "nervous"]
        },
        {
            "input": "I just got promoted at work and I'm so excited!",
            "expected_keywords": ["wonderful", "excited", "congratulations", "proud", "achievement"]
        },
        {
            "input": "I'm feeling overwhelmed with all my responsibilities.",
            "expected_keywords": ["understand", "overwhelmed", "responsibilities", "help", "manage"]
        },
        {
            "input": "I'm really grateful for my friends and family.",
            "expected_keywords": ["grateful", "beautiful", "appreciate", "wonderful", "support"]
        },
        {
            "input": "I'm not sure what I want to do with my life.",
            "expected_keywords": ["common", "natural", "uncertain", "figure", "challenge"]
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. Testing: '{test_case['input']}'")
        
        try:
            start_time = time.time()
            response = model.generate_response(test_case['input'])
            generation_time = time.time() - start_time
            
            print(f"Response: {response}")
            print(f"Generation time: {generation_time:.2f}s")
            
            # Check for emotional intelligence indicators
            response_lower = response.lower()
            found_keywords = [keyword for keyword in test_case['expected_keywords'] 
                            if keyword in response_lower]
            
            if found_keywords:
                print(f"✅ Found emotional intelligence keywords: {found_keywords}")
            else:
                print(f"⚠️  Expected keywords not found: {test_case['expected_keywords']}")
                
        except Exception as e:
            print(f"❌ Response generation failed: {e}")

def test_chat_interface(model):
    """Test chat interface functionality"""
    print("\n🧪 Testing Chat Interface...")
    
    try:
        response = model.chat("Hello! How are you today?")
        print(f"Chat response: {response}")
        print("✅ Chat interface working!")
    except Exception as e:
        print(f"❌ Chat interface failed: {e}")

def test_generation_parameters(model):
    """Test custom generation parameters"""
    print("\n🧪 Testing Generation Parameters...")
    
    try:
        # Test with different parameters
        response1 = model.generate_response(
            "I'm feeling stressed.",
            temperature=0.5,
            max_new_tokens=100
        )
        print(f"Conservative response: {response1}")
        
        response2 = model.generate_response(
            "I'm feeling stressed.",
            temperature=0.9,
            max_new_tokens=200
        )
        print(f"Creative response: {response2}")
        
        print("✅ Generation parameters working!")
    except Exception as e:
        print(f"❌ Generation parameters failed: {e}")

def test_memory_efficiency():
    """Test memory efficiency"""
    print("\n🧪 Testing Memory Efficiency...")
    
    try:
        # Test with standard loading
        model_standard = BrelloEI0(
            model_path="microsoft/DialoGPT-medium",
            load_in_4bit=False
        )
        
        # Get model size info
        if hasattr(model_4bit.model, 'get_memory_footprint'):
            memory_footprint = model_4bit.model.get_memory_footprint()
            print(f"Model memory footprint: {memory_footprint / 1024**3:.2f} GB")
        
        print("✅ Memory efficiency test passed!")
        return model_4bit
    except Exception as e:
        print(f"❌ Memory efficiency test failed: {e}")
        return None

def main():
    """Run all tests"""
    print("🤖 Brello EI 0 - Test Suite")
    print("Created by Epic Systems | Engineered by Rehan Temkar")
    print("=" * 50)
    
    # Test model loading
    model = test_model_loading()
    
    if model is None:
        print("\n❌ Cannot proceed with tests - model loading failed")
        return
    
    # Run all tests
    test_emotional_intelligence_responses(model)
    test_chat_interface(model)
    test_generation_parameters(model)
    test_memory_efficiency()
    
    print("\n🎉 All tests completed!")
    print("\n💡 If you encounter any issues:")
    print("1. Make sure you have accepted the model license on Hugging Face")
    print("2. Check that all dependencies are installed: pip install -r requirements.txt")
    print("3. Ensure you have sufficient GPU memory (at least 4GB recommended)")

if __name__ == "__main__":
    main()
