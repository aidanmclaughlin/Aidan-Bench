from concurrent.futures import ThreadPoolExecutor, as_completed
from benchmark import benchmark_question
from colorama import Fore, Style
from itertools import product
from question_list import questions
from get_args import get_user_choices
import json
import sys
import time
import os


def run_benchmark(
    model_names: list[str],
    temperatures: list[float],
    chain_of_thought: bool = False,
    use_llm: bool = False,
    multithreaded: bool = True,
    num_questions: int | None = None,
    results_file: str = 'results.json',
    thresholds: dict = None
) -> None:
    questions_to_use = questions[:num_questions] if num_questions else questions

    # Create results file if it doesn't exist
    if not os.path.exists(results_file):
        with open(results_file, 'w') as f:
            json.dump({}, f)

    with open(results_file, 'r') as f:
        results = json.load(f)

    start_time = time.time()

    try:
        _run_benchmarks(
            questions_to_use,
            model_names,
            temperatures,
            chain_of_thought,
            use_llm,
            results,
            multithreaded,
            results_file,
            thresholds
        )
    except KeyboardInterrupt:
        print(
            f"\n{Fore.YELLOW}Benchmark interrupted. Saving results...{Style.RESET_ALL}")
    finally:
        _save_results(results, results_file)
        print(f"Total processing time: {time.time() - start_time:.2f} seconds")


def _validate_environment() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print(
            f"{Fore.RED}Error: Missing required environment variable: OPENAI_API_KEY{Style.RESET_ALL}")
        sys.exit(1)
    if not os.getenv("OPEN_ROUTER_KEY"):
        print(
            f"{Fore.RED}Error: Missing required environment variable: OPEN_ROUTER_KEY{Style.RESET_ALL}")
        sys.exit(1)


def _run_benchmarks(questions, models, temperatures, chain_of_thought, use_llm, results, multithreaded, results_file, thresholds):
    benchmark_params = list(product(questions, models, temperatures))

    # Group parameters by model for tracking completion
    model_params = {}
    for question, model, temp in benchmark_params:
        model_params.setdefault(model, []).append((question, model, temp))

    if multithreaded:
        _run_multithreaded(model_params, chain_of_thought,
                           use_llm, results, results_file, thresholds)
    else:
        _run_sequential(model_params, chain_of_thought,
                        use_llm, results, results_file, thresholds)


def _run_multithreaded(model_params, chain_of_thought, use_llm, results, results_file, thresholds):
    with ThreadPoolExecutor(max_workers=100) as executor:
        all_futures = []
        active_models = []

        # Submit all tasks first
        for model, model_tasks in model_params.items():
            # Check if all questions for this model can be skipped
            if all(_can_skip_question(results, question, model, temp, use_llm, thresholds)
                   for question, model, temp in model_tasks):
                print(
                    f"Skipping all questions for {model} - already completed")
                continue

            active_models.append(model)
            model_futures = [
                executor.submit(_process_question, question, model, temp,
                                chain_of_thought, use_llm, results, thresholds)
                for question, model, temp in model_tasks
            ]
            all_futures.extend(model_futures)

        # Process all futures as they complete
        completed = 0
        total = len(all_futures)

        if total == 0:
            print("No tasks to process - all models completed")
            return

        print(f"Processing {total} tasks across {len(active_models)} models")

        for future in as_completed(all_futures):
            try:
                future.result()
                completed += 1
                if completed % 10 == 0:  # Progress update every 10 tasks
                    print(f"Completed {completed}/{total} tasks")
            except Exception as e:
                print(f"{Fore.RED}Error during benchmark: {e}{Style.RESET_ALL}")
                completed += 1

            # Save results periodically
            if completed % 50 == 0 or completed == total:
                _save_results(results, results_file)


def _run_sequential(model_params, chain_of_thought, use_llm, results, results_file, thresholds):
    for model_tasks in model_params.values():
        for question, model, temp in model_tasks:
            try:
                _process_question(question, model, temp,
                                  chain_of_thought, use_llm, results, thresholds)
            except Exception as e:
                print(
                    f"{Fore.RED}Error for {model} (temp={temp}): {e}{Style.RESET_ALL}")
        # Save results after each model completes
        _save_results(results, results_file)


def _process_question(question, model_name, temperature, chain_of_thought, use_llm, results, thresholds):
    # Get the model's results dict, creating nested structure if needed
    model_results = results.setdefault('models', {}).setdefault(
        model_name, {}).setdefault(str(temperature), {})

    # Skip if question is already completed successfully
    if question in model_results:
        previous_answers = model_results[question]
        if _should_skip_question(previous_answers, use_llm, thresholds):
            return

    # Get previous answers if they exist, otherwise empty list
    previous_answers = model_results.get(question, [])

    new_answers = benchmark_question(
        question,
        model_name,
        temperature,
        [a['answer'] for a in previous_answers],
        chain_of_thought,
        use_llm,
        thresholds
    )

    # Store results
    model_results[question] = previous_answers + new_answers


def _should_skip_question(previous_answers: list[dict], use_llm: bool, thresholds: dict) -> bool:
    if not previous_answers:
        return False

    last_answer = previous_answers[-1]

    checks = [
        last_answer.get('coherence_score',
                        100) <= thresholds['coherence_score'],
        last_answer.get('embedding_dissimilarity_score',
                        1.0) <= thresholds['embedding_dissimilarity_score']
    ]

    if use_llm:
        checks.append(
            last_answer.get('llm_dissimilarity_score',
                            1.0) <= thresholds['llm_dissimilarity_score']
        )

    return any(checks)


def _save_results(results: dict, results_file: str) -> None:
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)


def _can_skip_question(results: dict, question: str, model_name: str, temperature: float, use_llm: bool, thresholds: dict) -> bool:
    """Check if a question can be skipped before creating a thread for it"""
    model_results = (results.get('models', {})
                     .get(model_name, {})
                     .get(str(temperature), {}))

    previous_answers = model_results.get(question, [])
    return _should_skip_question(previous_answers, use_llm, thresholds) if previous_answers else False


if __name__ == "__main__":
    try:
        _validate_environment()
        choices = get_user_choices()
        run_benchmark(**choices)
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Benchmark interrupted. Exiting...{Style.RESET_ALL}")
        sys.exit(0)
