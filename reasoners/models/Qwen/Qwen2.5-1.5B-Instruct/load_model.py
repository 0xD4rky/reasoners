import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

@dataclass 
class QuantizationConfig:
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    bnb_4bit_compute_dtype: str = "bfloat16"  
    bnb_4bit_quant_type: str = "nf4" 
    bnb_4bit_use_double_quant: bool = True

@dataclass
class LoraConfig:
    rank: int = 32
    lora_alpha: int = 64
    lora_attn_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "output_proj"])
    lora_dropout: float = 0.01
    lora_bias: str = "none"
    apply_to_mlp: bool = True

    def get_target_modules(self) -> List[str]:
        modules = self.lora_attn_modules.copy()
        if self.apply_to_mlp:
            modules.extend(["gate_proj", "up_proj", "down_proj"])
        return modules


    
    

