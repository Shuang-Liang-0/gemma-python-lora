"""
Evaluation script for systematic model assessment.

Usage:
    python evaluate.py --adapter-path ./outputs/lora_r16 --num-samples 100
"""

import argparse
import json
import ast
from typing import Dict, List
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
from sacrebleu import sentence_bleu
from inference import load_model, generate_code


def check_syntax(code: str) -> Dict:
    """Check if code is syntactically valid Python."""
    try:
        ast.parse(code)
        return {'is_valid': True, 'error_type': None, 'error_msg': None}
    except SyntaxError as e:
        return {'is_valid': False, 'error_type': 'SyntaxError', 'error_msg': str(e)}
    except Exception as e:
        return {'is_valid': False, 'error_type': type(e).__name__, 'error_msg': str(e)}


def calculate_bleu(generated: str, reference: str) -> float:
    """Calculate BLEU score."""
    score = sentence_bleu(generated, [reference])
    return score.score


def extract_instruction_and_reference(example_text: str) -> tuple[str, str]:
    """
    Extract instruction and reference code from formatted example.
    """
    # Parse formatted text
    parts = example_text.split('<end_of_turn>')
    instruction = parts[0].replace('<start_of_turn>user\n', '').strip()
    reference = parts[1].replace('<start_of_turn>model\n', '').strip() if len(parts) > 1 else ""
    return instruction, reference


def evaluate_model(
    model,
    tokenizer,
    test_dataset,
    num_samples: int = 100,
    output_path: str = None
) -> Dict:
    """
    Comprehensively evaluate model.

    Returns:
        Dictionary of metrics and results
    """
    print(f"\nEvaluating on {num_samples} samples...")

    results = []
    syntax_valid = 0
    bleu_scores = []
    error_types = {}

    # Sample test examples
    test_samples = test_dataset.shuffle(seed=42).select(range(min(num_samples, len(test_dataset))))

    for i, example in enumerate(tqdm(test_samples)):
        # Extract instruction and reference
        instruction, reference = extract_instruction_and_reference(example['text'])

        # Generate
        generated = generate_code(
            model,
            tokenizer,
            instruction,
            max_length=256,
            temperature=0.7
        )

        # Syntax check
        syntax_check = check_syntax(generated)
        if syntax_check['is_valid']:
            syntax_valid += 1
        else:
            error_type = syntax_check['error_type']
            error_types[error_type] = error_types.get(error_type, 0) + 1

        # BLEU score
        bleu = calculate_bleu(generated, reference)
        bleu_scores.append(bleu)

        # Store result
        results.append({
            'id': i,
            'instruction': instruction,
            'reference': reference,
            'generated': generated,
            'syntax_valid': syntax_check['is_valid'],
            'syntax_error': syntax_check['error_type'],
            'bleu': bleu,
        })

    # Aggregate metrics
    metrics = {
        'num_samples': num_samples,
        'syntax_accuracy': syntax_valid / num_samples,
        'syntax_accuracy_pct': (syntax_valid / num_samples) * 100,
        'bleu_score_mean': float(np.mean(bleu_scores)),
        'bleu_score_std': float(np.std(bleu_scores)),
        'bleu_score_median': float(np.median(bleu_scores)),
        'error_breakdown': error_types,
        'avg_code_length': float(np.mean([len(r['generated'].split()) for r in results])),
    }

    # Print results
    print("\n" + "="*80)
    print("Evaluation Results")
    print("="*80)
    print(f"Samples evaluated: {metrics['num_samples']}")
    print(f"Syntax accuracy: {metrics['syntax_accuracy_pct']:.2f}%")
    print(f"BLEU score: {metrics['bleu_score_mean']:.2f} ± {metrics['bleu_score_std']:.2f}")
    print(f"Median BLEU: {metrics['bleu_score_median']:.2f}")
    print(f"Avg code length: {metrics['avg_code_length']:.1f} tokens")

    if error_types:
        print(f"\nError breakdown:")
        for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  {error_type}: {count} ({count/num_samples*100:.1f}%)")

    # Save results
    if output_path:
        output_data = {
            'metrics': metrics,
            'results': results
        }
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    return metrics, results


def main():
    parser = argparse.ArgumentParser(description="Evaluate code generation model")
    parser.add_argument(
        "--base-model",
        type=str,
        default="google/gemma-2-2b",
        help="Base model name"
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Path to LoRA adapter"
    )
    parser.add_argument(
        "--use-4bit",
        action="store_true",
        help="Load in 4-bit"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="iamtarun/python_code_instructions_18k_alpaca",
        help="Dataset name"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples to evaluate"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.json",
        help="Output file for results"
    )

    args = parser.parse_args()

    # Load model
    print("Loading model...")
    model, tokenizer = load_model(
        base_model_name=args.base_model,
        adapter_path=args.adapter_path,
        use_4bit=args.use_4bit
    )

    # Load dataset
    print(f"Loading dataset: {args.dataset}...")
    dataset = load_dataset(args.dataset)

    # Format dataset
    def format_instruction(example):
        instruction = example.get('instruction', '')
        input_text = example.get('input', '')
        output_text = example.get('output', '')

        user_message = instruction
        if input_text:
            user_message += f"\n{input_text}"

        formatted = f"""<start_of_turn>user
{user_message}<end_of_turn>
<start_of_turn>model
{output_text}<end_of_turn>"""
        return {'text': formatted}

    formatted_dataset = dataset['train'].map(format_instruction)
    test_dataset = formatted_dataset.train_test_split(test_size=0.1, seed=42)['test']

    # Evaluate
    metrics, results = evaluate_model(
        model,
        tokenizer,
        test_dataset,
        num_samples=args.num_samples,
        output_path=args.output
    )

    print("\n" + "="*80)
    print("Evaluation complete!")
    print("="*80)


if __name__ == "__main__":
    main()
