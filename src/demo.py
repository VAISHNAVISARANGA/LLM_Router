
"""Interactive demo for the trained LLM router.

Examples:
    python -m src.demo --dry-run --query "Explain Q-learning versus DQN"
    python -m src.demo --query "What is 17*23?"
    python -m src.demo --compare --query "Explain gradient descent"
    python -m src.demo --compare-all --query "Explain gradient descent"
"""

import argparse
import os

from dotenv import load_dotenv

from .router import LLMRouter


def tier_labels(n_models):
    if n_models == 2:
        return ["SMALL", "LARGE"]
    if n_models >= 3:
        middle = (
            ["MEDIUM"]
            if n_models == 3
            else [f"MEDIUM_{i + 1}" for i in range(n_models - 2)]
        )
        return ["SMALL", *middle, "LARGE"]
    raise ValueError(
        f"Expected at least 2 models in router metadata; found {n_models}"
    )


def get_price_config(label):
    """Return input/output prices per million tokens, or None if unconfigured."""
    in_key = f"{label}_PRICE_IN"
    out_key = f"{label}_PRICE_OUT"

    if in_key not in os.environ or out_key not in os.environ:
        return None

    try:
        return float(os.environ[in_key]), float(os.environ[out_key])
    except ValueError as exc:
        raise ValueError(
            f"{in_key} and {out_key} must be numeric."
        ) from exc


def price(label, usage):
    """Calculate cost from actual token usage and configured prices."""
    rates = get_price_config(label)
    if rates is None or usage is None:
        return None

    input_rate, output_rate = rates
    return (
        usage.prompt_tokens * input_rate
        + usage.completion_tokens * output_rate
    ) / 1_000_000


def ask(client, model, prompt, max_tokens):
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or "", response.usage


def show_scores(q_values, labels, selected):
    """Display the dual-adjusted action scores."""
    print("\nDual-adjusted action scores (not quality predictions):")
    print(f"{'Model tier':<12} {'Action':>7} {'Score':>12}  Status")
    print("-" * 48)

    for i, score in enumerate(q_values):
        status = "<-- SELECTED" if i == selected else ""
        print(f"{labels[i]:<12} {i:>7} {float(score):>12.5f}  {status}")

    print(
        f"\nDecision: {labels[selected]} had the highest adjusted score among "
        f"{len(q_values)} actions."
    )


def print_cost(label, model, cost):
    if cost is None:
        print(
            f"[{label} / {model}] Cost unavailable: configure "
            f"{label}_PRICE_IN and {label}_PRICE_OUT in .env."
        )
    else:
        print(f"[{label} / {model}] Measured-token cost: ${cost:.6f}")


def main():
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query",
        help="A single prompt. If omitted, read prompts from --file.",
    )
    parser.add_argument("--file", default="demo_queries.txt")
    parser.add_argument("--agent", choices=["q", "dqn"], default=None)
    parser.add_argument(
        "--state", choices=["handcrafted", "embedding", "hybrid"], default=None
    )
    parser.add_argument("--models-dir", default="models")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show routing decisions without calling model APIs.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Call the selected model and compare it with the large model.",
    )
    parser.add_argument(
        "--compare-all",
        action="store_true",
        help="Call every model and display each response and measured-token cost.",
    )
    parser.add_argument("--max-tokens", type=int, default=300)
    args = parser.parse_args()

    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive.")

    if args.compare and args.compare_all:
        parser.error("Use either --compare or --compare-all, not both.")

    router = LLMRouter(
        models_dir=args.models_dir,
        agent=args.agent,
        state=args.state,
    )

    model_ids = router.meta["models"]
    labels = tier_labels(len(model_ids))

    if args.query is not None:
        queries = [args.query]
    else:
        with open(args.file, encoding="utf-8") as file:
            queries = [line.strip() for line in file if line.strip()]

    if not queries:
        parser.error("No prompts found. Supply --query or a non-empty --file.")

    # Map each tier to its configured provider model name, if overridden.
    names = {
        label: os.getenv(f"{label}_MODEL_NAME", model_id)
        for label, model_id in zip(labels, model_ids)
    }

    client = None
    if not args.dry_run:
        from openai import OpenAI

        api_key = os.getenv("ROUTER_API_KEY")
        if not api_key:
            parser.error(
                "ROUTER_API_KEY is missing. Set it in .env or use --dry-run."
            )

        client = OpenAI(
            api_key=api_key,
            base_url=os.getenv("ROUTER_BASE_URL") or None,
        )

    routed_costs = []
    large_costs = []
    large_count = 0

    for prompt in queries:
        action, scores = router.route(prompt)
        selected_label = labels[action]
        large_label = labels[-1]

        print("\n" + "=" * 76)
        print(f"PROMPT: {prompt}")
        show_scores(scores, labels, action)
        print(f"Current dual lambda: {router.dual_lambda:.5f}")

        if action == len(labels) - 1:
            large_count += 1

        if client is None:
            print(
                "\nDry run: no API calls made; no response quality or live cost "
                "measured, so the dual price is not updated."
            )
            continue

        # Compare every model, or the selected model plus the large baseline.
        if args.compare_all:
            call_labels = labels
        elif args.compare:
            call_labels = list(dict.fromkeys([selected_label, large_label]))
        else:
            call_labels = [selected_label]

        results = {}

        for label in call_labels:
            model_name = names[label]
            try:
                answer, usage = ask(
                    client, model_name, prompt, args.max_tokens
                )
            except Exception as exc:
                print(f"\n[{label} / {model_name}] API call failed: {exc}")
                continue

            cost = price(label, usage)
            results[label] = {"answer": answer, "cost": cost}

            print(f"\n--- {label} ({model_name}) ---")
            print(answer)
            print_cost(label, model_name, cost)

        if selected_label in results:
            selected_cost = results[selected_label]["cost"]
            routed_costs.append(selected_cost)
            router.observe_cost(selected_cost)
        else:
            router.observe_cost(None)

        if large_label in results:
            large_costs.append(results[large_label]["cost"])

        if args.compare_all:
            print(
                "\nNote: responses are shown side by side; this demo does not "
                "automatically determine which answer is best."
            )

    print("\n" + "=" * 76)
    print(f"Prompts processed: {len(queries)}")
    print(f"Prompts routed to large model: {large_count}/{len(queries)}")

    if client is not None and args.compare:
        valid_routed = [x for x in routed_costs if x is not None]
        valid_large = [x for x in large_costs if x is not None]

        if (
            len(valid_routed) == len(queries)
            and len(valid_large) == len(queries)
            and sum(valid_large) > 0
        ):
            routed_total = sum(valid_routed)
            large_total = sum(valid_large)
            saved = 100 * (1 - routed_total / large_total)
            print(
                f"Selected-model cost: ${routed_total:.6f} | "
                f"Always-large cost: ${large_total:.6f} | "
                f"Cost difference: {saved:.1f}%"
            )
        else:
            print(
                "Cost comparison unavailable or incomplete. Check API usage "
                "and configure input/output prices for both tiers."
            )


if __name__ == "__main__":
    main()
