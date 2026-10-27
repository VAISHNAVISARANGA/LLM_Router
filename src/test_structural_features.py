from src.features import extract_structural_features

prompts = [
    "What is 2 + 2?",
    "Explain this Python code:\n```python\nprint('hello')\n```",
    r"Derive the equation $\frac{a+b}{c}=x$ step by step.",
]

for prompt in prompts:
    features = extract_structural_features(prompt)
    print("\nPrompt:", prompt)
    print("Features:", features)
    print("Feature count:", len(features))