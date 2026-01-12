import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Path to your base noeum model
MODEL_PATH = "./base/Noeum-hf-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    print(f"--- Evaluating Base Model Noeum on {DEVICE} ---")

    # 1. Load Resources
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    ).to(DEVICE)
    model.eval()

    # Helper function for generation
    def run_test(test_name, prompt, max_new=50, temp=0.7):
        print(f"\n=== {test_name} ===")
        print(f"Input Pattern:\n{prompt.strip()}")
        print("-" * 20)

        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=True,
                temperature=temp,
                top_p=0.9,
                use_cache=False,  # Essential for your MoE architecture compatibility
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # Decode only the NEW tokens to see exactly what the model added
        new_tokens = output_ids[0][inputs.input_ids.shape[1]:]
        output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        print(f"Model Completion:\n{output_text}")
        print("=" * 30)

    # ==============================================================================
    # TEST 1: Few-Shot Knowledge
    # Base models need examples to know they should answer, not ask more questions.
    # ==============================================================================
    few_shot_prompt = """
Q: What is the capital of Germany?
A: Berlin
Q: What is the capital of Spain?
A: Madrid
Q: What is the capital of France?
A:"""
    run_test("Test 1: Few-Shot Knowledge", few_shot_prompt, max_new=10, temp=0.1)

    # ==============================================================================
    # TEST 2: Story Continuation
    # Tests the model's ability to maintain narrative flow and grammar.
    # ==============================================================================
    story_prompt = "The spaceship landed silently on the unknown planet. The captain opened the hatch and saw"
    run_test("Test 2: Creative Writing", story_prompt, max_new=60, temp=0.8)

    # ==============================================================================
    # TEST 3: Logic/Code Pattern
    # Base models are often good at completing structured patterns or code.
    # ==============================================================================
    code_prompt = """
def add(a, b):
    return a + b

def multiply(a, b):"""
    run_test("Test 3: Code/Pattern Completion", code_prompt, max_new=30, temp=0.2)


if __name__ == "__main__":
    main()
