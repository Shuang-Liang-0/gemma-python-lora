# Usage Guide

This guide explains how to use the trained LoRA/QLoRA models and evaluation scripts.

## Table of Contents

1. [Training (Google Colab)](#training)
2. [Inference](#inference)
3. [Interactive Demo](#demo)
4. [Evaluation](#evaluation)
5. [Functional Testing](#functional-testing)

---

## Training (Google Colab)

### Step 1: Open Colab Notebook

1. Upload `notebook/gemma_lora_training.ipynb` to Google Colab
2. Or open directly: [Open in Colab](https://colab.research.google.com/github/YOUR-USERNAME/gemma-python-lora/blob/main/notebook/gemma_lora_training.ipynb)

### Step 2: Configure Runtime

- Go to `Runtime` → `Change runtime type`
- Select `T4 GPU` (free) or `A100 GPU` (Pro)
- Click `Save`

### Step 3: Run Training

- Run all cells in order
- Training takes ~2-4 hours for LoRA, ~3-5 hours for QLoRA
- Models are automatically saved to `./outputs/`

### Step 4: Download Trained Models

After training, download the adapter weights:

```python
# In Colab
from google.colab import files

# Download LoRA adapters
!zip -r lora_r16.zip ./outputs/lora_r16
files.download('lora_r16.zip')

# Download QLoRA adapters
!zip -r qlora_r16.zip ./outputs/qlora_r16
files.download('qlora_r16.zip')
```

---

## Inference

Use trained models to generate code from natural language instructions.

### Basic Usage

```bash
# Using LoRA adapter
python src/inference.py \
  --adapter-path ./outputs/lora_r16 \
  --prompt "Write a function to calculate fibonacci numbers"

# Using QLoRA adapter (4-bit)
python src/inference.py \
  --adapter-path ./outputs/qlora_r16 \
  --use-4bit \
  --prompt "Create a function to reverse a string"

# Using base model (no adapter)
python src/inference.py \
  --prompt "Write a function to check if a number is prime"
```

### Advanced Options

```bash
python src/inference.py \
  --adapter-path ./outputs/lora_r16 \
  --prompt "Your instruction here" \
  --max-length 512 \
  --temperature 0.7 \
  --top-p 0.9
```

Parameters:
- `--max-length`: Maximum tokens to generate (default: 256)
- `--temperature`: Sampling temperature, higher = more random (default: 0.7)
- `--top-p`: Nucleus sampling threshold (default: 0.9)

### Using from Python

```python
from src.inference import load_model, generate_code

# Load model
model, tokenizer = load_model(
    base_model_name="google/gemma-2-2b",
    adapter_path="./outputs/lora_r16",
    use_4bit=False
)

# Generate code
code = generate_code(
    model,
    tokenizer,
    "Write a function to calculate the factorial of a number",
    max_length=256,
    temperature=0.7
)

print(code)
```

---

## Interactive Demo

Run an interactive session for code generation.

### Start Demo

```bash
# With LoRA
python src/demo.py --adapter-path ./outputs/lora_r16

# With QLoRA
python src/demo.py --adapter-path ./outputs/qlora_r16 --use-4bit

# Base model only
python src/demo.py
```

### Demo Commands

Once running:
- Enter your instruction and press Enter
- Type `examples` to see example prompts
- Type `quit` or `exit` to stop

### Example Session

```
Enter instruction: Write a function to reverse a string

Generating code...

================================================================================
Generated Code:
================================================================================
def reverse_string(s):
    """Reverses the input string."""
    return s[::-1]
================================================================================

✓ Syntax check: PASSED

Enter instruction: quit
Goodbye!
```

---

## Evaluation

Systematically evaluate model performance with multiple metrics.

### Run Evaluation

```bash
# Evaluate LoRA model
python src/evaluate.py \
  --adapter-path ./outputs/lora_r16 \
  --num-samples 100 \
  --output lora_eval.json

# Evaluate QLoRA model
python src/evaluate.py \
  --adapter-path ./outputs/qlora_r16 \
  --use-4bit \
  --num-samples 100 \
  --output qlora_eval.json

# Evaluate base model
python src/evaluate.py \
  --num-samples 100 \
  --output base_eval.json
```

### Evaluation Metrics

The script computes:
- **Syntax Accuracy**: % of generated code that is syntactically valid
- **BLEU Score**: Overlap with reference code
- **Error Breakdown**: Types of syntax errors
- **Code Statistics**: Average length, complexity

### Sample Output

```
Evaluation Results
================================================================================
Samples evaluated: 100
Syntax accuracy: 95.00%
BLEU score: 42.35 ± 8.12
Median BLEU: 41.50
Avg code length: 48.2 tokens

Error breakdown:
  SyntaxError: 3 (3.0%)
  IndentationError: 2 (2.0%)
================================================================================
```

---

## Functional Testing

Test if generated code actually executes correctly (pass@1 metric).

### Run Functional Tests

```bash
# Test LoRA model
python src/functional_test.py \
  --adapter-path ./outputs/lora_r16 \
  --verbose \
  --output functional_results.json

# Test QLoRA model
python src/functional_test.py \
  --adapter-path ./outputs/qlora_r16 \
  --use-4bit \
  --output qlora_functional.json
```

### What It Tests

The script includes predefined test cases for common tasks:
1. Calculate factorial
2. Reverse string
3. Check if prime
4. Find max in list
5. Sum list of numbers

Each task has multiple test inputs/outputs to verify correctness.

### Sample Output

```
Task 1: Write a function to calculate the factorial of a number.

Generated code:
def factorial(n):
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)

✓ PASS | Input: 0 | Expected: 1 | Got: 1
✓ PASS | Input: 1 | Expected: 1 | Got: 1
✓ PASS | Input: 5 | Expected: 120 | Got: 120
✓ PASS | Input: 10 | Expected: 3628800 | Got: 3628800

Functional Correctness Results
================================================================================
Pass@1: 80.0% (4/5 tasks)
Test pass rate: 92.3% (18/20 tests)
================================================================================
```

---

## Loading from HuggingFace Hub

After uploading to HuggingFace, load models directly:

```python
from src.inference import load_model, generate_code

# Load LoRA from HuggingFace
model, tokenizer = load_model(
    base_model_name="google/gemma-2-2b",
    adapter_path="YOUR-USERNAME/gemma-2-2b-python-lora"
)

# Generate
code = generate_code(model, tokenizer, "Write a binary search function")
print(code)
```

Or from command line:

```bash
python src/inference.py \
  --adapter-path YOUR-USERNAME/gemma-2-2b-python-lora \
  --prompt "Write a binary search function"
```

---

## Tips and Best Practices

### For Inference

1. **Lower temperature (0.3-0.5)** for more deterministic, reliable code
2. **Higher temperature (0.7-0.9)** for more creative solutions
3. **Clear, specific instructions** get better results
4. **Use QLoRA** (4-bit) if you have limited VRAM (<8GB)

### For Evaluation

1. **Use 100+ samples** for statistically significant results
2. **Test on held-out data** not seen during training
3. **Compare multiple metrics** (syntax, BLEU, functional correctness)
4. **Manual inspection** of failures reveals improvement areas

### For Training

1. **Monitor syntax accuracy** during training as a key metric
2. **Use lower rank (r=8)** for faster training, higher (r=32) for quality
3. **QLoRA enables larger batch sizes** due to memory savings
4. **Save checkpoints frequently** in case of session interruptions

---

## Troubleshooting

### Out of Memory Errors

```bash
# Use 4-bit quantization
python src/inference.py --adapter-path PATH --use-4bit --prompt "..."

# Reduce batch size in training (edit notebook)
CONFIG['batch_size'] = 2  # Instead of 4

# Use gradient accumulation
CONFIG['gradient_accumulation_steps'] = 8  # Instead of 4
```

### Slow Generation

```bash
# Reduce max length
python src/inference.py --max-length 128 --prompt "..."

# Use greedy decoding (faster than sampling)
# Edit inference.py: do_sample=False
```

### Poor Code Quality

- Try lower temperature (0.3-0.5)
- Use more specific instructions
- Try higher rank model (r=32 vs r=16)
- Ensure proper training (check loss curves)

---

## Next Steps

- Upload models to HuggingFace Hub (see notebook section 10)
- Fine-tune on your own dataset
- Experiment with different ranks and hyperparameters
- Add your own test cases for functional testing
- Deploy as a web API using FastAPI/Flask

---

For more details, see the [README](README.md) or check the [notebook](notebook/gemma_lora_training.ipynb).
