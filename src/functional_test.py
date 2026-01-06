"""
Functional correctness testing (pass@1) for generated code.

This script tests if generated code actually executes correctly on test cases.

WARNING: This executes generated code, which can be unsafe. Only use on trusted outputs.

Usage:
    python functional_test.py --adapter-path ./outputs/lora_r16
"""

import argparse
import json
import ast
import sys
from io import StringIO
from contextlib import contextmanager
from typing import Dict, List, Any
from datasets import load_dataset
from tqdm import tqdm
from inference import load_model, generate_code


@contextmanager
def capture_stdout():
    """Context manager to capture stdout."""
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        yield sys.stdout
    finally:
        sys.stdout = old_stdout


def extract_function_name(code: str) -> str:
    """
    Extract the main function name from Python code.
    """
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                return node.name
    except:
        pass
    return None


def safe_execute(code: str, test_input: Any, timeout: int = 5) -> tuple[bool, Any, str]:
    """
    Safely execute generated code with test input.

    Returns:
        (success, output, error_message)
    """
    try:
        # Parse code to check syntax
        ast.parse(code)

        # Create execution environment
        exec_globals = {}
        exec_locals = {}

        # Execute code
        exec(code, exec_globals, exec_locals)

        # Find the main function
        func_name = extract_function_name(code)
        if not func_name:
            return False, None, "No function found in generated code"

        if func_name not in exec_locals:
            return False, None, f"Function '{func_name}' not found in execution context"

        # Call function with test input
        func = exec_locals[func_name]

        # Handle different input types
        if isinstance(test_input, (list, tuple)):
            result = func(*test_input)
        elif isinstance(test_input, dict):
            result = func(**test_input)
        else:
            result = func(test_input)

        return True, result, None

    except SyntaxError as e:
        return False, None, f"SyntaxError: {e}"
    except TimeoutError:
        return False, None, "Execution timeout"
    except Exception as e:
        return False, None, f"{type(e).__name__}: {e}"


def create_test_cases() -> List[Dict]:
    """
    Create test cases for common programming tasks.

    Returns list of test case dictionaries with:
    - instruction: Task description
    - test_inputs: List of inputs to test
    - expected_outputs: List of expected outputs
    """
    test_cases = [
        {
            'instruction': 'Write a function to calculate the factorial of a number.',
            'test_inputs': [0, 1, 5, 10],
            'expected_outputs': [1, 1, 120, 3628800],
        },
        {
            'instruction': 'Create a function to reverse a string.',
            'test_inputs': ['hello', 'python', 'a', ''],
            'expected_outputs': ['olleh', 'nohtyp', 'a', ''],
        },
        {
            'instruction': 'Write a function to check if a number is prime.',
            'test_inputs': [2, 3, 4, 17, 100],
            'expected_outputs': [True, True, False, True, False],
        },
        {
            'instruction': 'Create a function to find the maximum element in a list.',
            'test_inputs': [[1, 2, 3], [5, 1, 9, 3], [-1, -5, -2]],
            'expected_outputs': [3, 9, -1],
        },
        {
            'instruction': 'Write a function to calculate the sum of a list of numbers.',
            'test_inputs': [[1, 2, 3], [10, 20, 30], [], [5]],
            'expected_outputs': [6, 60, 0, 5],
        },
    ]
    return test_cases


def test_model_functional_correctness(
    model,
    tokenizer,
    test_cases: List[Dict] = None,
    verbose: bool = True
) -> Dict:
    """
    Test functional correctness of generated code.

    Args:
        model: Model to test
        tokenizer: Tokenizer
        test_cases: List of test case dictionaries
        verbose: Print detailed results

    Returns:
        Dictionary with pass rates and details
    """
    if test_cases is None:
        test_cases = create_test_cases()

    print(f"\nTesting functional correctness on {len(test_cases)} tasks...\n")

    results = []
    total_tests = 0
    passed_tests = 0
    tasks_passed = 0

    for task_id, task in enumerate(tqdm(test_cases)):
        instruction = task['instruction']
        test_inputs = task['test_inputs']
        expected_outputs = task['expected_outputs']

        if verbose:
            print(f"\n{'='*80}")
            print(f"Task {task_id + 1}: {instruction}")
            print('='*80)

        # Generate code
        generated_code = generate_code(
            model,
            tokenizer,
            instruction,
            max_length=256,
            temperature=0.3  # Lower temperature for more deterministic output
        )

        if verbose:
            print(f"\nGenerated code:\n{generated_code}\n")

        # Test on all inputs
        task_results = []
        task_passed = True

        for test_input, expected_output in zip(test_inputs, expected_outputs):
            total_tests += 1

            # Execute
            success, actual_output, error = safe_execute(generated_code, test_input)

            # Check correctness
            is_correct = success and actual_output == expected_output

            if is_correct:
                passed_tests += 1
                status = "✓ PASS"
            else:
                task_passed = False
                status = "✗ FAIL"

            task_results.append({
                'input': test_input,
                'expected': expected_output,
                'actual': actual_output,
                'success': success,
                'is_correct': is_correct,
                'error': error
            })

            if verbose:
                print(f"{status} | Input: {test_input} | Expected: {expected_output} | Got: {actual_output}")
                if error:
                    print(f"  Error: {error}")

        if task_passed:
            tasks_passed += 1

        results.append({
            'task_id': task_id,
            'instruction': instruction,
            'generated_code': generated_code,
            'test_results': task_results,
            'task_passed': task_passed
        })

    # Calculate metrics
    pass_at_1 = tasks_passed / len(test_cases)
    test_pass_rate = passed_tests / total_tests if total_tests > 0 else 0

    metrics = {
        'pass_at_1': pass_at_1,
        'pass_at_1_pct': pass_at_1 * 100,
        'tasks_passed': tasks_passed,
        'total_tasks': len(test_cases),
        'test_pass_rate': test_pass_rate,
        'test_pass_rate_pct': test_pass_rate * 100,
        'tests_passed': passed_tests,
        'total_tests': total_tests,
    }

    # Print summary
    print("\n" + "="*80)
    print("Functional Correctness Results")
    print("="*80)
    print(f"Pass@1: {metrics['pass_at_1_pct']:.1f}% ({tasks_passed}/{len(test_cases)} tasks)")
    print(f"Test pass rate: {metrics['test_pass_rate_pct']:.1f}% ({passed_tests}/{total_tests} tests)")
    print("="*80)

    return metrics, results


def main():
    parser = argparse.ArgumentParser(description="Test functional correctness of generated code")
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
        "--output",
        type=str,
        default="functional_test_results.json",
        help="Output file"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed results"
    )

    args = parser.parse_args()

    # Load model
    print("Loading model...")
    model, tokenizer = load_model(
        base_model_name=args.base_model,
        adapter_path=args.adapter_path,
        use_4bit=args.use_4bit
    )

    # Run tests
    metrics, results = test_model_functional_correctness(
        model,
        tokenizer,
        verbose=args.verbose
    )

    # Save results
    output_data = {
        'metrics': metrics,
        'results': results
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
