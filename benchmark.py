from prompts import *
from models import embed
from colorama import Fore, Style
import time
import numpy as np
import pickle
from filelock import FileLock
import concurrent.futures

judge_model = "anthropic/claude-3.5-sonnet"


def benchmark_question(question: str, model_name: str, temperature: float, chain_of_thought: bool = False, use_llm: bool = False):
    start_time = time.time()
    results = _load_results()
    _ensure_result_structure(results, model_name, temperature, question)
    model_results = results['models'][model_name][temperature]
    previous_answers = [data['answer'] for data in model_results[question]]
    answer_num = len(previous_answers) + 1

    total_novelty_score = 0.0

    while True:
        try:
            new_answer = gen_answer(
                question,
                previous_answers,
                model_name,
                chain_of_thought
            )
            coherence_score = judge_answer(question, new_answer, judge_model)

            if coherence_score <= 3:
                print(
                    f"{Fore.YELLOW}Output is incoherent. Stopping generation for this question.{Style.RESET_ALL}")
                break

            novelty_score = _check_similarity(
                question, new_answer, previous_answers, use_llm
            )
            # If using llm, num matches is K⋅N + 2K(K−1), where K is len(prev_answers) and N how many answers we've generated
            total_novelty_score += novelty_score  # Accumulate the novelty score

            if novelty_score < 0.1:
                print(
                    f"{Fore.YELLOW}Output is redundant. Stopping generation for this question.{Style.RESET_ALL}")
                break

            _save_answer(results, model_results, question, new_answer, novelty_score,
                         coherence_score, answer_num, start_time)
            previous_answers.append(new_answer)  # Update previous answers
            answer_num += 1

        except Exception as e:
            print(f"{Fore.RED}Error processing question: {str(e)}{Style.RESET_ALL}")
            break

    print(f"Total Novelty Score: {total_novelty_score}")


def _ensure_result_structure(results: dict, model_name: str, temperature: float, question: str):
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

    model_results[question].append(answer_data)
    _save_results(results)

    print(
        f"{Fore.CYAN}Question: {question}{Style.RESET_ALL}\n"
        f"{Fore.GREEN}Answer #{answer_num}: {answer}{Style.RESET_ALL}\n"
        f"{Fore.MAGENTA}Coherence Score: {coherence_score}{Style.RESET_ALL}\n"
        f"{Fore.BLUE}Dissimilarity Score: {novelty_score:.2f}{Style.RESET_ALL}\n"
        f"{Fore.YELLOW}Processing Time: {answer_data['processing_time']:.2f} seconds{Style.RESET_ALL}\n\n"
    )


def _save_results(results: dict, filename: str = 'results.pkl'):
    lockfile = f"{filename}.lock"
    with FileLock(lockfile):
        with open(filename, 'wb') as f:
            pickle.dump(results, f)


def _check_similarity(question: str, new_answer: str, previous_answers: list, use_llm: bool) -> float:
    if not previous_answers:
        return 1.0

    if use_llm:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(previous_answers)) as executor:
            similarities = list(executor.map(
                lambda prev_answer: judge_similarity(question, new_answer, prev_answer, judge_model),
                previous_answers
            ))
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
    lockfile = f"{filename}.lock"
    try:
        with FileLock(lockfile):
            with open(filename, 'rb') as f:
                return pickle.load(f)
    except FileNotFoundError:
        return {'models': {}}
