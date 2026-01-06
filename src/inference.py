"""
Inference script for loading and using trained LoRA/QLoRA models.

Usage:
    python inference.py --prompt "Write a function to calculate fibonacci numbers"
    python inference.py --model-path ./outputs/lora_r16 --prompt "Your instruction here"
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


def load_model(
    base_model_name: str = "google/gemma-2-2b",
    adapter_path: str = None,
    use_4bit: bool = False,
    device_map: str = "auto"
):
    """
    Load base model and optionally merge LoRA adapters.

    Args:
        base_model_name: HuggingFace model ID for base model
        adapter_path: Path to LoRA adapter weights (local or HuggingFace)
        use_4bit: Whether to load in 4-bit (for QLoRA models)
        device_map: Device mapping strategy

    Returns:
        model, tokenizer
    """
    print(f"Loading tokenizer from {base_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    if use_4bit:
        print("Loading base model in 4-bit...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True
        )
    else:
        print("Loading base model in FP16...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map=device_map,
            trust_remote_code=True
        )

    if adapter_path:
        print(f"Loading LoRA adapters from {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)
        print("LoRA adapters loaded successfully!")

    model.eval()
    return model, tokenizer


def generate_code(
    model,
    tokenizer,
    instruction: str,
    max_length: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True
) -> str:
    """
    Generate Python code from instruction.

    Args:
        model: The model to use for generation
        tokenizer: Tokenizer for the model
        instruction: Natural language instruction
        max_length: Maximum tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_p: Nucleus sampling threshold
        do_sample: Whether to use sampling (vs greedy)

    Returns:
        Generated code as string
    """
    # Format prompt in Gemma instruction format
    prompt = f"""<start_of_turn>user
{instruction}<end_of_turn>
<start_of_turn>model
"""

    # Tokenize
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the model's response
    if '<start_of_turn>model' in generated_text:
        generated_code = generated_text.split('<start_of_turn>model')[-1].strip()
    else:
        generated_code = generated_text[len(prompt):].strip()

    return generated_code


def main():
    parser = argparse.ArgumentParser(description="Generate Python code using LoRA/QLoRA fine-tuned Gemma")
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Instruction for code generation"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="google/gemma-2-2b",
        help="Base model name or path"
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Path to LoRA adapter (local path or HuggingFace model ID)"
    )
    parser.add_argument(
        "--use-4bit",
        action="store_true",
        help="Load model in 4-bit (for QLoRA adapters)"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling threshold"
    )

    args = parser.parse_args()

    # Load model
    model, tokenizer = load_model(
        base_model_name=args.base_model,
        adapter_path=args.adapter_path,
        use_4bit=args.use_4bit
    )

    # Generate
    print("\n" + "="*80)
    print(f"Instruction: {args.prompt}")
    print("="*80)
    print("\nGenerating code...\n")

    generated_code = generate_code(
        model,
        tokenizer,
        args.prompt,
        max_length=args.max_length,
        temperature=args.temperature,
        top_p=args.top_p
    )

    print("Generated Code:")
    print("-"*80)
    print(generated_code)
    print("="*80)


if __name__ == "__main__":
    main()
