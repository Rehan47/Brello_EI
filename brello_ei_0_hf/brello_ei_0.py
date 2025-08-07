"""
Brello EI 0 - Emotional Intelligence AI Model
Created by Epic Systems | Engineered by Rehan Temkar

A locally-run emotional intelligence AI model based on Llama 3.2 3B,
designed to provide empathetic, emotionally-aware responses.
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    BitsAndBytesConfig
)
from typing import Optional, Dict, Any, List
import logging
import os

logger = logging.getLogger(__name__)

class BrelloEI0:
    """
    Brello EI 0 - Emotional Intelligence AI Model
    
    A locally-run AI model designed to provide emotionally intelligent,
    empathetic responses with natural conversation flow.
    """
    
    def __init__(
        self,
        model_path: str = "microsoft/DialoGPT-medium",
        device: Optional[str] = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        torch_dtype: Optional[torch.dtype] = None,
        **kwargs
    ):
        """
        Initialize Brello EI 0 model
        
        Args:
            model_path: Path to Llama 3.2 3B model
            device: Device to load model on ('cuda', 'cpu', etc.)
            load_in_4bit: Whether to load model in 4-bit quantization
            load_in_8bit: Whether to load model in 8-bit quantization
            torch_dtype: Torch data type for model weights
        """
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.config = {
            "max_length": 4096,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "do_sample": True,
            "min_length": 30,
            "max_new_tokens": 256,
            "no_repeat_ngram_size": 3
        }
        
        # Quantization config for memory efficiency
        self.quantization_config = None
        if load_in_4bit:
            self.quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif load_in_8bit:
            self.quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        
        self.torch_dtype = torch_dtype or torch.float16 if self.device == "cuda" else torch.float32
        
        logger.info(f"Initializing Brello EI 0 model: {model_path}")
        self.load_model()
    
    def load_model(self):
        """Load the Brello EI 0 model and tokenizer"""
        try:
            logger.info(f"Loading Brello EI 0 model: {self.model_path}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                padding_side="left"
            )
            
            # Set padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            model_kwargs = {
                "torch_dtype": self.torch_dtype,
                "device_map": "auto" if self.device == "cuda" else None,
                "trust_remote_code": True
            }
            
            if self.quantization_config:
                model_kwargs["quantization_config"] = self.quantization_config
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **model_kwargs
            )
            
            # Move to device if not using device_map
            if self.device != "cuda" or self.quantization_config is None:
                self.model = self.model.to(self.device)
            
            logger.info("✅ Brello EI 0 model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load Brello EI 0 model: {e}")
            raise
    
    def apply_emotional_intelligence_prompt(self, user_input: str) -> str:
        """
        Apply emotional intelligence prompt template for Brello EI 0
        
        Args:
            user_input: User's message
            
        Returns:
            Formatted conversation string with emotional intelligence focus
        """
        # Format the conversation with emotional intelligence focus
        prompt = f"""<|system|>
You are Brello EI 0, an emotionally intelligent AI created by Epic Systems and engineered by Rehan Temkar. You provide empathetic, understanding responses that show emotional awareness and genuine care for the user's feelings and experiences. You are part of the Brello AI family, designed to bring emotional intelligence to AI conversations.
</s>
<|user|>
{user_input}
</s>
<|assistant|>"""
        return prompt
    
    def generate_response(
        self,
        user_input: str,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        Generate emotionally intelligent response
        
        Args:
            user_input: User's message
            max_length: Maximum response length
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            **kwargs: Additional generation parameters
            
        Returns:
            Generated emotionally intelligent response
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Apply emotional intelligence prompt template
        formatted_input = self.apply_emotional_intelligence_prompt(user_input)
        
        # Tokenize input
        inputs = self.tokenizer.encode(formatted_input, return_tensors="pt")
        if hasattr(self.model, 'device'):
            inputs = inputs.to(self.model.device)
        
        # Generation parameters - optimized for emotional intelligence
        gen_params = {
            "max_length": max_length or self.config["max_length"],
            "temperature": temperature or self.config["temperature"],
            "top_p": top_p or self.config["top_p"],
            "do_sample": self.config["do_sample"],
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": self.config["repetition_penalty"],
            "length_penalty": 1.0,
            "no_repeat_ngram_size": self.config["no_repeat_ngram_size"],
            "min_length": self.config["min_length"],
            "max_new_tokens": self.config["max_new_tokens"],
            **kwargs
        }
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                **gen_params
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant's response
        if "<|assistant|>" in response:
            response = response.split("<|assistant|>")[-1].strip()
        
        # Clean up the response
        response = response.strip()
        
        # Ensure response shows emotional intelligence
        if len(response) < 20:
            response = f"I understand how you might be feeling. {response} It's important to acknowledge our emotions and experiences."
        
        return response
    
    def chat(self, message: str, maintain_history: bool = False) -> str:
        """
        Simple chat interface
        
        Args:
            message: User message
            maintain_history: Whether to maintain conversation history
            
        Returns:
            Model response
        """
        return self.generate_response(message)
    
    def __call__(self, text: str, **kwargs) -> str:
        """Convenience method for generating responses"""
        return self.generate_response(text, **kwargs)

# Convenience function for quick usage
def load_brello_ei_0(model_path: str = "microsoft/DialoGPT-medium", **kwargs) -> BrelloEI0:
    """
    Load Brello EI 0 model
    
    Args:
        model_path: Path to Llama 3.2 3B model
        **kwargs: Additional model parameters
        
    Returns:
        BrelloEI0 instance
    """
    return BrelloEI0(model_path=model_path, **kwargs)
