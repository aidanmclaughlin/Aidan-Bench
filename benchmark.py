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
    total_coherence_score = 0.0
    new_answers_data = []

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
                print(f"Output is incoherent. Stopping generation for this question.")
                print(f"Final Statistics:")
                print(f"Total Answers Generated: {len(new_answers_data)}")
                print(f"Total Embedding Novelty Score: {total_novelty_score:.2f}")
                print(f"Total Coherence Score: {total_coherence_score:.2f}")
                break

            novelty_scores = _check_similarity(
                question, new_answer, previous_answers, use_llm
            )
            embedding_novelty_score = novelty_scores['embedding_novelty_score']
            total_novelty_score += embedding_novelty_score  # Accumulate the embedding novelty score

            if embedding_novelty_score < 0.1:
                print(f"Output is redundant based on embedding similarity. Stopping generation for this question.")
                print(f"Final Statistics:")
                print(f"Total Answers Generated: {len(new_answers_data)}")
                print(f"Total Embedding Novelty Score: {total_novelty_score:.2f}")
                print(f"Total Coherence Score: {total_coherence_score:.2f}")
                break

            total_coherence_score += coherence_score

            answer_data = {
                'answer_num': answer_num,
                'answer': new_answer,
                'embedding_dissimilarity_score': embedding_novelty_score,
                'coherence_score': coherence_score,
                'processing_time': time.time() - start_time
            }

            if use_llm:
                llm_novelty_score = novelty_scores['llm_novelty_score']
                answer_data['llm_dissimilarity_score'] = llm_novelty_score

                if llm_novelty_score < 0.1:
                    print(f"Output is redundant based on LLM similarity. Stopping generation for this question.")
                    print(f"Final Statistics:")
                    print(f"Total Answers Generated: {len(new_answers_data)}")
                    print(f"Total Embedding Novelty Score: {total_novelty_score:.2f}")
                    print(f"Total Coherence Score: {total_coherence_score:.2f}")
                    if use_llm:
                        print(f"Total LLM Novelty Score: {sum(d.get('llm_dissimilarity_score', 0) for d in new_answers_data):.2f}")
                    break

            new_answers_data.append(answer_data)

            print(
                f"Using {model_name} with temperature {temperature}\n"
                f"{Fore.CYAN}Question: {question}{Style.RESET_ALL}\n"
                f"{Fore.GREEN}Answer #{answer_num}: {new_answer}{Style.RESET_ALL}\n"
                f"{Fore.MAGENTA}Coherence Score: {coherence_score}{Style.RESET_ALL}\n"
                f"{Fore.BLUE}Embedding Dissimilarity Score: {embedding_novelty_score:.2f}{Style.RESET_ALL}"
            )
            if use_llm:
                print(f"{Fore.BLUE}LLM Dissimilarity Score: {llm_novelty_score:.2f}{Style.RESET_ALL}")

            previous_answers.append(new_answer)  # Update previous answers
            answer_num += 1

        except Exception as e:
            print(f"{Fore.RED}Error processing question: {str(e)}{Style.RESET_ALL}")
            break

    if new_answers_data:
        model_results[question].extend(new_answers_data)
        _save_results(results)

    print(f"Total Embedding Novelty Score: {total_novelty_score}")


def _ensure_result_structure(results: dict, model_name: str, temperature: float, question: str):
    if model_name not in results['models']:
        results['models'][model_name] = {}
    if temperature not in results['models'][model_name]:
        results['models'][model_name][temperature] = {}
    if question not in results['models'][model_name][temperature]:
        results['models'][model_name][temperature][question] = []


def _save_results(results: dict, filename: str = 'novelty_model_test.pkl'):
    lockfile = f"{filename}.lock"
    with FileLock(lockfile):
        with open(filename, 'wb') as f:
            pickle.dump(results, f)


def _check_similarity(question: str, new_answer: str, previous_answers: list, use_llm: bool) -> dict:
    similarity_scores = {}

    if not previous_answers:
        similarity_scores['embedding_novelty_score'] = 1.0
        if use_llm:
            similarity_scores['llm_novelty_score'] = 1.0
        return similarity_scores

    # Calculate embedding-based novelty score
    embedding_novelty_score = _get_novelty_score(new_answer, previous_answers)
    similarity_scores['embedding_novelty_score'] = embedding_novelty_score

    # If use_llm is True, calculate LLM-based novelty score
    if use_llm:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(previous_answers)) as executor:
            similarities = list(executor.map(
                lambda prev_answer: judge_similarity(question, new_answer, prev_answer, judge_model),
                previous_answers
            ))
        llm_novelty_score = 1 - max(similarities)
        similarity_scores['llm_novelty_score'] = llm_novelty_score

    return similarity_scores


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


def _load_results(filename: str = 'novelty_model_test.pkl') -> dict:
    lockfile = f"{filename}.lock"
    try:
        with FileLock(lockfile):
            with open(filename, 'rb') as f:
                return pickle.load(f)
    except FileNotFoundError:
        return {'models': {}}
