import pickle
import numpy as np
from models import chat_with_model, embed
from prompts import questions, create_gen_prompt, create_judge_prompt
from colorama import Fore, Style
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
from threading import Lock


def benchmark_model(model_name: str, multithreaded: bool = False, temperature: float = 0.7, chain_of_thought: bool = False):
    global _results_lock
    _results_lock = Lock()
    results = _load_results()

    if multithreaded:
        with ThreadPoolExecutor(max_workers=len(questions)) as executor:
            futures = [
                executor.submit(process_question, q, model_name,
                                temperature, results, chain_of_thought)
                for q in questions
            ]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"{Fore.RED}Error in thread: {str(e)}{Style.RESET_ALL}")
    else:
        for question in questions:
            process_question(question, model_name, temperature,
                             results, chain_of_thought)

    return results


def process_question(question: str, model_name: str, temperature: float, results: dict, chain_of_thought: bool = False):
    start_time = time.time()

    _ensure_result_structure(results, model_name, temperature, question)
    model_results = results['models'][model_name][temperature]
    previous_answers = []
    answer_num = len(model_results[question]) + 1

    try:
        new_answer = chat_with_model(
            prompt=create_gen_prompt(
                question, previous_answers, chain_of_thought),
            model=model_name,
            temperature=temperature
        )

        coherence_score = _get_coherence_score(question, new_answer)
        if coherence_score <= 3:
            print(
                f"{Fore.YELLOW}Output is incoherent. Moving to next question.{Style.RESET_ALL}")
            return

        novelty_score = _get_novelty_score(new_answer, previous_answers)
        if novelty_score < 0.1:
            print(
                f"{Fore.YELLOW}Output is redundant. Moving to next question.{Style.RESET_ALL}")
            return

        _save_answer(results, model_results, question, new_answer, novelty_score,
                     coherence_score, answer_num, start_time)

    except Exception as e:
        print(f"{Fore.RED}Error processing question: {str(e)}{Style.RESET_ALL}")


def _ensure_result_structure(results: dict, model_name: str, temperature: float, question: str):
    with _results_lock:
        if model_name not in results['models']:
            results['models'][model_name] = {}
        if temperature not in results['models'][model_name]:
            results['models'][model_name][temperature] = {}
        if question not in results['models'][model_name][temperature]:
            results['models'][model_name][temperature][question] = []


def _get_coherence_score(question: str, answer: str) -> int:
    judge_response = chat_with_model(
        prompt=create_judge_prompt(question, answer), model="gpt-4o-mini")
    return int(judge_response.split("<coherence_score>")[1].split("</coherence_score>")[0])


def _get_novelty_score(new_answer: str, previous_answers: list) -> float:
    if not previous_answers:
        return 1.0

    new_embedding = embed(new_answer)
    previous_embeddings = [embed(answer) for answer in previous_answers]

    similarities = [np.dot(new_embedding, prev_embedding) /
                    (np.linalg.norm(new_embedding) *
                     np.linalg.norm(prev_embedding))
                    for prev_embedding in previous_embeddings]

    return 1 - max(similarities)


def _save_answer(results: dict, model_results: dict, question: str, answer: str,
                 novelty_score: float, coherence_score: int, answer_num: int, start_time: float):
    answer_data = {
        'answer_num': answer_num,
        'answer': answer,
        'dissimilarity_score': novelty_score,
        'coherence_score': coherence_score,
        'processing_time': time.time() - start_time
    }

    with _results_lock:
        model_results[question].append(answer_data)
        _save_results(results)


def _load_results(filename: str = 'results.pkl') -> dict:
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {
            'models': {}  # model_name -> {temp -> {question -> [answer_data]}}
        }


def _save_results(results: dict, filename: str = 'results.pkl'):
    with _results_lock:
        with open(filename, 'wb') as f:
            pickle.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark a language model.")
    parser.add_argument("model_name", type=str,
                        help="Name of the model to benchmark")
    parser.add_argument("--single-threaded", action="store_true",
                        help="Run in single-threaded mode")
    parser.add_argument("--temperature", type=float,
                        default=0.7, help="Temperature for generation")
    parser.add_argument("--chain-of-thought", action="store_true",
                        help="Enable chain-of-thought prompting")
    args = parser.parse_args()

    benchmark_model(args.model_name, not args.single_threaded,
                    args.temperature, args.chain_of_thought)
