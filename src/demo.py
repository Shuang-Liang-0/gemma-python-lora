"""
Interactive demo for Python code generation with LoRA/QLoRA models.

Usage:
    python demo.py
    python demo.py --adapter-path ./outputs/lora_r16
"""

import argparse
import ast
from inference import load_model, generate_code


def check_syntax(code: str) -> tuple[bool, str]:
    """
    Check if generated code is syntactically valid.

    Returns:
        (is_valid, error_message)
    """
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def interactive_demo(model, tokenizer):
    """
    Run interactive code generation demo.
    """
    print("\n" + "="*80)
    print("  Gemma-2-2B Python Code Generator (LoRA/QLoRA)")
    print("="*80)
    print("\nInstructions:")
    print("  - Enter a natural language instruction for code generation")
    print("  - Type 'quit' or 'exit' to stop")
    print("  - Type 'examples' to see example prompts")
    print("="*80)

    example_prompts = [
        "Write a function to calculate the factorial of a number.",
        "Create a function to reverse a string.",
        "Write a function to check if a number is prime.",
        "Create a function to merge two sorted lists.",
        "Write a function to find the nth Fibonacci number.",
        "Create a class to implement a binary search tree.",
        "Write a function to calculate the greatest common divisor.",
        "Create a function to validate an email address using regex.",
    ]

    while True:
        print("\n" + "-"*80)
        user_input = input("Enter instruction (or 'quit'): ").strip()

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break

        if user_input.lower() == 'examples':
            print("\nExample prompts:")
            for i, example in enumerate(example_prompts, 1):
                print(f"  {i}. {example}")
            continue

        if not user_input:
            print("Please enter a valid instruction.")
            continue

        # Generate code
        print("\nGenerating code...")
        generated_code = generate_code(
            model,
            tokenizer,
            user_input,
            max_length=256,
            temperature=0.7
        )

        # Display result
        print("\n" + "="*80)
        print("Generated Code:")
        print("="*80)
        print(generated_code)
        print("="*80)

        # Check syntax
        is_valid, error = check_syntax(generated_code)
        if is_valid:
            print("\n✓ Syntax check: PASSED")
        else:
            print(f"\n✗ Syntax check: FAILED")
            print(f"  Error: {error}")


def main():
    parser = argparse.ArgumentParser(description="Interactive demo for code generation")
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
        help="Path to LoRA adapter (local or HuggingFace)"
    )
    parser.add_argument(
        "--use-4bit",
        action="store_true",
        help="Load model in 4-bit (for QLoRA)"
    )

    args = parser.parse_args()

    # Load model
    print("Loading model...")
    model, tokenizer = load_model(
        base_model_name=args.base_model,
        adapter_path=args.adapter_path,
        use_4bit=args.use_4bit
    )
    print("Model loaded successfully!\n")

    # Run demo
    interactive_demo(model, tokenizer)


if __name__ == "__main__":
    main()
