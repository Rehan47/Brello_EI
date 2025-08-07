# Brello EI 0 - Emotional Intelligence AI

A conversational AI model designed for emotional intelligence and empathetic responses.

## Features

- Emotional Intelligence: Designed for empathetic, understanding responses
- Conversational: Natural dialogue flow and engagement
- Local Operation: Can run completely locally
- Memory Efficient: Optimized for limited hardware
- Easy Integration: Simple API for quick integration

## Installation

```bash
pip install torch transformers accelerate
```

## Quick Start

```python
from brello_ei_0 import BrelloEI0

# Load the model
model = BrelloEI0(
    model_path="microsoft/DialoGPT-medium",
    load_in_4bit=False
)

# Generate emotionally intelligent response
response = model.generate_response("I'm feeling really stressed about my presentation tomorrow.")
print(response)
```

## Example Conversations

```python
# Anxiety Support
response = model.generate_response("I'm feeling really anxious about my job interview tomorrow.")

# Celebrating Success
response = model.generate_response("I just got promoted at work!")

# Emotional Support
response = model.generate_response("I'm feeling lonely and isolated.")
```

## Use Cases

### Emotional Support
- Providing empathetic responses to stress and anxiety
- Supporting users through difficult life transitions
- Celebrating achievements and successes

### Personal Development
- Career guidance and decision-making support
- Life goal exploration and planning
- Self-reflection and emotional awareness

### Mental Health Support
- Stress management and coping strategies
- Emotional validation and understanding
- Positive reinforcement and encouragement

## Configuration

### Model Parameters
- `model_path`: Path to base model (default: "microsoft/DialoGPT-medium")
- `device`: Device to load model on ('cuda', 'cpu', etc.)
- `load_in_4bit`: Enable 4-bit quantization for memory efficiency
- `load_in_8bit`: Enable 8-bit quantization for memory efficiency

### Generation Parameters
- `temperature`: Sampling temperature (default: 0.7)
- `top_p`: Top-p sampling parameter (default: 0.9)
- `max_length`: Maximum response length (default: 4096)
- `min_length`: Minimum response length (default: 30)
- `max_new_tokens`: Maximum new tokens to generate (default: 256)

## Performance

### Model Specifications
- Foundation: Microsoft DialoGPT-medium
- Parameters: 345 Million
- Context Length: 1024 tokens
- Training: Conversational dialogue data
- Optimization: Emotional intelligence focus

### Memory Requirements
- Full Precision: ~1GB VRAM
- 8-bit Quantization: ~500MB VRAM
- 4-bit Quantization: ~250MB VRAM (recommended)

## Architecture

Brello EI 0 is built on advanced language model architecture with the following key components:

- Base Model: Microsoft DialoGPT-medium
- Tokenizer: Optimized for conversational data
- Generation: Emotionally intelligent response patterns
- Post-processing: Response cleaning and enhancement
- Quantization: 4-bit for memory efficiency (optional)

## Contributing

For questions or contributions, please contact the development team.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Microsoft for the DialoGPT foundation model
- Hugging Face for the transformers library
