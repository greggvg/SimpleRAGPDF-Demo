"""LLM loading and inference utilities"""

import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class LLMGenerator:
    """Manages LLM model and generation"""
    
    def __init__(self, model_name: str, device: str = None):
        """
        Load LLM model and tokenizer.
        
        Args:
            model_name: Name of the HuggingFace model
            device: Device to load model on ('cuda', 'cpu'). Auto-detects if None.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = torch.device(device)
        self.model_name = model_name
        
        print(f"Loading tokenizer for {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print(f"Loading model {model_name}...")
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Set padding token if not set
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"Model loaded on: {next(self.model.parameters()).device}")
    
    def generate(
        self,
        prompt: str,
        max_input_tokens: int = 512,
        max_new_tokens: int = 120,
        do_sample: bool = False
    ) -> tuple:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input prompt
            max_input_tokens: Maximum input length
            max_new_tokens: Maximum new tokens to generate
            do_sample: Whether to use sampling
            
        Returns:
            Tuple of (generated text, timing dict)
        """
        t0 = time.time()
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        t1 = time.time()
        
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                do_sample=do_sample,
                max_new_tokens=max_new_tokens,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        t2 = time.time()
        
        # Extract only the newly generated tokens
        gen_ids = outputs[0][inputs["input_ids"].shape[-1]:]
        answer = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        
        timing = {
            "tokenize": round(t1 - t0, 3),
            "generate": round(t2 - t1, 3),
            "total": round(t2 - t0, 3),
            "input_tokens": inputs["input_ids"].shape[-1]
        }
        
        return answer, timing
