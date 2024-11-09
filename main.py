from colorama import Fore, Style
from concurrent.futures import ThreadPoolExecutor, as_completed
from benchmark import benchmark_question
from model_list import models
from question_list import questions
import argparse


def benchmark_model(model_name: str, multithreaded: bool = False, temperature: float | None = 0.7, chain_of_thought: bool = False):
    if isinstance(model_name, list):
        _benchmark_multiple_models(
            model_name, multithreaded, temperature, chain_of_thought)
        return

    if isinstance(temperature, list):
        _benchmark_temperature_range(
            model_name, multithreaded, temperature, chain_of_thought)
        return

    if multithreaded:
        with ThreadPoolExecutor(max_workers=len(questions)) as executor:
            futures = [
                executor.submit(benchmark_question, q, model_name,
                                temperature, chain_of_thought)
                for q in questions
            ]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"{Fore.RED}Error in thread: {str(e)}{Style.RESET_ALL}")
    else:
        for question in questions:
            benchmark_question(question, model_name,
                               temperature, chain_of_thought)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark a language model.")
    parser.add_argument("model_name", type=str, nargs='?',
                        help="Name of the model to benchmark")
    parser.add_argument("--all-models", action="store_true",
                        help="Benchmark all available models")
    parser.add_argument("--temp-range", action="store_true",
                        help="Benchmark with temperatures from 0 to 1 in 0.1 increments")
    parser.add_argument("--single-threaded", action="store_true",
                        help="Run in single-threaded mode")
    parser.add_argument("--temperature", type=float,
                        default=0.7, help="Temperature for generation")
    parser.add_argument("--chain-of-thought", action="store_true",
                        help="Enable chain-of-thought prompting")
    args = parser.parse_args()

    if args.all_models and args.model_name:
        parser.error("Cannot specify both model_name and --all-models")

    model_name = models if args.all_models else args.model_name
    temperature = [round(t * 0.1, 1) for t in range(11)
                   ] if args.temp_range else args.temperature

    if not args.all_models and not args.model_name:
        parser.error("Must specify either model_name or --all-models")

    benchmark_model(model_name, not args.single_threaded,
                    temperature, args.chain_of_thought)


def _benchmark_multiple_models(models: list[str], multithreaded: bool, temperature: float | list[float], chain_of_thought: bool):
    if multithreaded:
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(benchmark_model, model, False,
                                temperature, chain_of_thought)
                for model in models
            ]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"{Fore.RED}Error in thread: {str(e)}{Style.RESET_ALL}")
    else:
        for model in models:
            benchmark_model(model, False, temperature, chain_of_thought)


def _benchmark_temperature_range(model: str, multithreaded: bool, temperatures: list[float], chain_of_thought: bool):
    if multithreaded:
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(benchmark_model, model,
                                False, temp, chain_of_thought)
                for temp in temperatures
            ]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"{Fore.RED}Error in thread: {str(e)}{Style.RESET_ALL}")
    else:
        for temp in temperatures:
            benchmark_model(model, False, temp, chain_of_thought)
