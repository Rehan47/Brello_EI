# Brello EI 0 - Emotional Intelligence AI Model

**Created by Epic Systems | Engineered by Rehan Temkar**

A locally-run emotional intelligence AI model designed to provide empathetic, emotionally-aware responses with natural conversation flow.

## Features

- **Emotional Intelligence**: Designed to provide empathetic, understanding responses
- **Local Operation**: Runs completely locally without external dependencies
- **Memory Efficient**: 4-bit quantization for optimal performance on limited hardware
- **Advanced Architecture**: Based on Llama 3.2 3B foundation model
- **Easy Integration**: Simple API for quick integration
- **Flexible Configuration**: Customizable generation parameters

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended) or CPU
- At least 8GB RAM (16GB recommended)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Model Options

**Option 1: Use Public Model (Recommended for quick start)**
The default configuration uses `microsoft/DialoGPT-medium` which is publicly available and doesn't require authentication.

**Option 2: Use Llama 3.2 3B (Requires authentication)**
To use the actual Llama 3.2 3B model:
1. Create a Hugging Face account
2. Accept the model license at: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
3. Login with: `huggingface-cli login` or `hf auth login`
4. Update the model_path in your code to: `"meta-llama/Llama-3.2-3B-Instruct"`

**Option 3: Use Other Public Models**
- `microsoft/DialoGPT-large` (larger, better responses)
- `microsoft/DialoGPT-small` (faster, smaller)
- `HuggingFaceTB/SmolLM3-3B` (3B parameter model)

## Quick Start

### Basic Usage

```python
from brello_ei_0 import BrelloEI0

# Load the model
model = BrelloEI0(
    model_path="microsoft/DialoGPT-medium",  # Public model, no auth required
    load_in_4bit=False  # Set to True if you have CUDA
)

# Generate an emotionally intelligent response
response = model.generate_response("I'm feeling really stressed about my job interview.")
print(response)
```

### Alternative Loading

```python
from brello_ei_0 import load_brello_ei_0

# Load model using convenience function
model = load_brello_ei_0("microsoft/DialoGPT-medium")

# Direct call
response = model("I'm really happy about my recent success!")
print(response)
```

### Chat Interface

```python
# Simple chat
response = model.chat("How are you feeling today?")
print(response)
```

## 🎮 Example Conversations

```python
# Example 1: Anxiety Support
response = model.generate_response("I'm feeling really anxious about my presentation tomorrow.")
# Output: "I can understand how nerve-wracking presentations can be. It's completely natural to feel anxious..."

# Example 2: Celebrating Success
response = model.generate_response("I just got promoted at work!")
# Output: "That's wonderful! I can feel your excitement and it's absolutely contagious..."

# Example 3: Emotional Support
response = model.generate_response("I'm feeling lonely and isolated.")
# Output: "I'm so sorry you're feeling this way. Loneliness can be really painful..."

# Example 4: Career Guidance
response = model.generate_response("I'm confused about what I want to do with my life.")
# Output: "That's a really common and natural feeling, especially when we're at crossroads..."
```

## ⚙️ Configuration

### Model Parameters

- `model_path`: Path to Llama 3.2 3B model (default: "meta-llama/Meta-Llama-3.2-3B-Instruct")
- `device`: Device to load model on ('cuda', 'cpu', etc.)
- `load_in_4bit`: Enable 4-bit quantization for memory efficiency (recommended)
- `load_in_8bit`: Enable 8-bit quantization for memory efficiency
- `torch_dtype`: Torch data type for model weights

### Generation Parameters

- `temperature`: Sampling temperature (default: 0.7)
- `top_p`: Top-p sampling parameter (default: 0.9)
- `max_length`: Maximum response length (default: 4096)
- `min_length`: Minimum response length (default: 30)
- `max_new_tokens`: Maximum new tokens to generate (default: 256)
- `repetition_penalty`: Penalty for repetition (default: 1.1)

## 📊 Performance

### Model Specifications

- **Foundation**: Microsoft DialoGPT-medium
- **Parameters**: 345 Million
- **Context Length**: 1024 tokens
- **Training**: Conversational dialogue data
- **Optimization**: Emotional intelligence focus

### Memory Requirements

- **Full Precision**: ~1GB VRAM
- **8-bit Quantization**: ~500MB VRAM
- **4-bit Quantization**: ~250MB VRAM (recommended)

## 🔧 Advanced Usage

### Custom Generation Parameters

```python
response = model.generate_response(
    "I'm feeling overwhelmed with my responsibilities.",
    temperature=0.8,
    top_p=0.95,
    max_new_tokens=300,
    repetition_penalty=1.05
)
```

### Batch Processing

```python
messages = [
    "I'm really proud of my accomplishments.",
    "I'm feeling uncertain about my future.",
    "I'm grateful for my support system."
]

responses = []
for message in messages:
    response = model.generate_response(message)
    responses.append(response)
```

## Training

### Fine-tune for Emotional Intelligence

```bash
python train_brello_ei_0.py
```

The training script will:
- Load Llama 3.2 3B with 4-bit quantization
- Apply LoRA for efficient fine-tuning
- Train on emotional intelligence data
- Save the fine-tuned model

### Training Data

The model is fine-tuned on emotional intelligence scenarios:
- Anxiety and stress support
- Celebrating success and achievements
- Dealing with loneliness and isolation
- Career guidance and life decisions
- Gratitude and appreciation
- Overwhelm and responsibility management

## Architecture

Brello EI 0 is built on advanced language model architecture with the following key components:

- **Base Model**: Microsoft DialoGPT-medium
- **Tokenizer**: Optimized for conversational data
- **Generation**: Emotionally intelligent response patterns
- **Post-processing**: Response cleaning and enhancement
- **Quantization**: 4-bit for memory efficiency (optional)

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

## Contributing

This model is part of the Epic Systems AI initiative. For questions or contributions, please contact the development team.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **Epic Systems** for the vision and support
- **Rehan Temkar** for engineering and development
- **Microsoft** for the DialoGPT foundation model
- **Hugging Face** for the transformers library

---

**Brello EI 0** - Bringing emotional intelligence to AI conversations 💙✨
