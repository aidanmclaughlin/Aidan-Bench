from colorama import Fore, Style
import os
from model_list import models, model_subset

DEFAULT_THRESHOLDS = {
    'coherence_score': 15,
    'embedding_dissimilarity_score': 0.15,
    'llm_dissimilarity_score': 0.15
}

SECTION_COLOR = Fore.MAGENTA + Style.BRIGHT
PROMPT_COLOR = Fore.CYAN
ERROR_COLOR = Fore.RED + Style.BRIGHT

def _clear_screen():
    os.system('clear' if os.name != 'nt' else 'cls')

def get_user_choices() -> dict[str, any]:
    choices = {}
    
    _clear_screen()
    print(f"\n{SECTION_COLOR}=== LLM Benchmark Configuration ==={Style.RESET_ALL}\n")
    
    choices['model_names'] = _get_model_selection()
    _clear_screen()
    
    choices['thresholds'] = _get_threshold_configuration()
    _clear_screen()
    
    choices['temperatures'] = _get_temperature_configuration()
    _clear_screen()
    
    print(f"{SECTION_COLOR}=== Additional Configuration ==={Style.RESET_ALL}\n")
    choices['chain_of_thought'] = _get_yes_no(f"{PROMPT_COLOR}Use chain of thought?{Style.RESET_ALL}")
    choices['use_llm'] = _get_yes_no(f"{PROMPT_COLOR}Use LLM similarity scoring?{Style.RESET_ALL}")
    choices['multithreaded'] = _get_yes_no(f"{PROMPT_COLOR}Enable multithreading?{Style.RESET_ALL}")
    choices['num_questions'] = _get_num_questions()
    choices['results_file'] = _get_results_file()
    
    return choices

def _get_model_selection() -> list[str]:
    """Handle model selection logic."""
    print("Select models to benchmark:")
    print("1. Single model")
    print("2. Model subset")
    print("3. All models")

    while True:
        model_choice = input("\nEnter choice (1-3): ").strip()
        if model_choice in {'1', '2', '3'}:
            break
        print(f"{ERROR_COLOR}Invalid choice. Please enter 1, 2, or 3.{Style.RESET_ALL}")

    if model_choice == '1':
        print("\nAvailable models:")
        for idx, model in enumerate(models, 1):
            print(f"{idx}. {model}")
        while True:
            try:
                model_idx = int(input("\nEnter model number: ").strip()) - 1
                if 0 <= model_idx < len(models):
                    return [models[model_idx]]
            except ValueError:
                pass
            print(f"{ERROR_COLOR}Invalid choice. Please enter a number between 1 and {len(models)}.{Style.RESET_ALL}")
    elif model_choice == '2':
        return model_subset
    return models

def _get_threshold_configuration() -> dict:
    """Handle threshold configuration."""
    print(f"\n{SECTION_COLOR}Skip Thresholds Configuration{Style.RESET_ALL}")
    print("Current defaults:")
    for key, value in DEFAULT_THRESHOLDS.items():
        print(f"- {key}: {value}")

    if not _get_yes_no("\nModify thresholds?"):
        return DEFAULT_THRESHOLDS.copy()

    thresholds = {}
    print("\nEnter new values (press Enter to keep default):")
    
    thresholds['coherence_score'] = _get_float_or_default(
        "Coherence score threshold (lower is better)",
        DEFAULT_THRESHOLDS['coherence_score'],
        0, 100
    )
    thresholds['embedding_dissimilarity_score'] = _get_float_or_default(
        "Embedding dissimilarity threshold (lower is better)",
        DEFAULT_THRESHOLDS['embedding_dissimilarity_score'],
        0, 1
    )
    thresholds['llm_dissimilarity_score'] = _get_float_or_default(
        "LLM dissimilarity threshold (lower is better)",
        DEFAULT_THRESHOLDS['llm_dissimilarity_score'],
        0, 1
    )
    
    return thresholds

def _get_temperature_configuration() -> list[float]:
    """Handle temperature configuration."""
    print("\nTemperature configuration:")
    print("1. Single temperature")
    print("2. Full range (0.0 - 1.0)")

    while True:
        temp_choice = input("\nEnter choice (1-2): ").strip()
        if temp_choice in {'1', '2'}:
            break
        print(f"{ERROR_COLOR}Invalid choice. Please enter 1 or 2.{Style.RESET_ALL}")

    if temp_choice == '1':
        while True:
            try:
                temp = float(input("\nEnter temperature (0.0 - 1.0): ").strip())
                if 0 <= temp <= 1:
                    return [temp]
            except ValueError:
                pass
            print(f"{ERROR_COLOR}Invalid temperature. Please enter a number between 0.0 and 1.0.{Style.RESET_ALL}")
    return [round(t * 0.2, 1) for t in range(6)]

def _get_num_questions() -> int | None:
    """Get number of questions to process."""
    while True:
        num_q = input("\nNumber of questions (press Enter for all): ").strip()
        if not num_q:
            return None
        try:
            num = int(num_q)
            if num > 0:
                return num
        except ValueError:
            pass
        print(f"{ERROR_COLOR}Invalid number. Please enter a positive integer or press Enter.{Style.RESET_ALL}")

def _get_results_file() -> str:
    """Get results file path."""
    return input("\nResults file path (press Enter for 'results.json'): ").strip() or 'results.json'

def _get_float_or_default(prompt: str, default: float, min_val: float, max_val: float) -> float:
    """Get float input with validation and default value."""
    while True:
        response = input(f"{prompt} (default={default}): ").strip()
        if not response:
            return default
        try:
            value = float(response)
            if min_val <= value <= max_val:
                return value
            print(f"{ERROR_COLOR}Value must be between {min_val} and {max_val}.{Style.RESET_ALL}")
        except ValueError:
            print(f"{ERROR_COLOR}Please enter a valid number.{Style.RESET_ALL}")

def _get_yes_no(prompt: str) -> bool:
    """Get yes/no input with validation."""
    while True:
        response = input(f"{prompt} (y/n): ").strip().lower()
        if response in {'y', 'yes'}:
            return True
        if response in {'n', 'no'}:
            return False
        print(f"{ERROR_COLOR}Please enter 'y' or 'n'.{Style.RESET_ALL}")
