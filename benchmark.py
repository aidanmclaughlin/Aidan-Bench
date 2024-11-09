from prompts import *
from models import embed
from colorama import Fore, Style
import time
import numpy as np
import pickle
from threading import Lock

_results_lock = Lock()


def benchmark_question(question: str, model_name: str, temperature: float, chain_of_thought: bool = False):
    start_time = time.time()
    
    results = _load_results()
    _ensure_result_structure(results, model_name, temperature, question)
    model_results = results['models'][model_name][temperature]
    previous_answers = [data['answer'] for data in model_results[question]]
    answer_num = len(previous_answers) + 1

    try:
        new_answer = gen_answer(question, previous_answers, chain_of_thought)

        coherence_score = judge_answer(question, new_answer)
        if coherence_score <= 3:
            print(
                f"{Fore.YELLOW}Output is incoherent. Moving to next question.{Style.RESET_ALL}")
            return

        novelty_score = _check_similarity(
            question, new_answer, previous_answers, False)
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


def _save_results(results: dict, filename: str = 'results.pkl'):
    with _results_lock:
        with open(filename, 'wb') as f:
            pickle.dump(results, f)


def _check_similarity(question: str, new_answer: str, previous_answers: list, use_llm: bool) -> float:
    if not previous_answers:
        return 1.0

    if use_llm:
        similarities = [judge_similarity(question, new_answer, prev_answer)
                        for prev_answer in previous_answers]
        return 1 - max(similarities)
    else:
        return _get_novelty_score(new_answer, previous_answers)


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


def _load_results(filename: str = 'results.pkl') -> dict:
    try:
        with _results_lock:
            with open(filename, 'rb') as f:
                return pickle.load(f)
    except FileNotFoundError:
        return {'models': {}}
