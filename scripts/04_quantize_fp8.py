"""
Task 3: FP8 dynamic quantization of Qwen/Qwen3-8B with llmcompressor.
Run inside comp_venv:
    source comp_venv/bin/activate
    python scripts/04_quantize_fp8.py
Output model saved to ./Qwen3-8B-FP8-Dynamic  (original BF16 model is untouched).
"""

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-8B"
OUTPUT_DIR = "Qwen3-8B-FP8-Dynamic"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",
    ignore=["lm_head"],
)

oneshot(
    model=model,
    recipe=recipe,
    output_dir=OUTPUT_DIR,
)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Quantized model saved → {OUTPUT_DIR}")
print("Verify quantization config:")
import json, pathlib
cfg = json.loads((pathlib.Path(OUTPUT_DIR) / "config.json").read_text())
print(json.dumps(cfg.get("quantization_config", {}), indent=2))
