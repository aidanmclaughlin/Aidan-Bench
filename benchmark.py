from prompts import *
import time
import concurrent.futures
import numpy as np
from colorama import Fore, Style
from models import embed


def benchmark_question(
    question: str,
    model_name: str,
    temperature: float,
    previous_answers: list,
    chain_of_thought: bool = False,
    use_llm: bool = False,
    thresholds: dict = None
):
    start_time = time.time()
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
            coherence_score = judge_answer(
                question, new_answer, model_name='o1-mini'
            )

            novelty_scores = _check_similarity(
                question, new_answer, previous_answers, use_llm
            )
            embedding_novelty_score = novelty_scores['embedding_novelty_score']
            total_novelty_score += embedding_novelty_score

            if use_llm:
                llm_novelty_score = novelty_scores['llm_novelty_score']

            total_coherence_score += coherence_score

            answer_data = {
                'answer_num': answer_num,
                'answer': new_answer,
                'embedding_dissimilarity_score': embedding_novelty_score,
                'coherence_score': coherence_score,
                'processing_time': time.time() - start_time
            }

            if use_llm:
                answer_data['llm_dissimilarity_score'] = llm_novelty_score

            new_answers_data.append(answer_data)
            previous_answers.append(new_answer)

            print(
                f"Using {model_name} with temperature {temperature}\n"
                f"{Fore.CYAN}Question: {question}{Style.RESET_ALL}\n"
                f"{Fore.GREEN}Answer #{answer_num}: {new_answer}{Style.RESET_ALL}\n"
                f"{Fore.MAGENTA}Coherence Score: {coherence_score}{Style.RESET_ALL}\n"
                f"{Fore.BLUE}Embedding Dissimilarity Score: {embedding_novelty_score:.2f}{Style.RESET_ALL}\n"
                f"{f'{Fore.BLUE}LLM Dissimilarity Score: {llm_novelty_score:.2f}{Style.RESET_ALL}' if use_llm else ''}\n"
            )

            answer_num += 1

            if (coherence_score <= thresholds['coherence_score'] or 
                embedding_novelty_score < thresholds['embedding_dissimilarity_score'] or
                (use_llm and llm_novelty_score < thresholds['llm_dissimilarity_score'])):
                print(f"Breaking after {answer_num} answers.")
                break

        except Exception as e:
            print(f"{Fore.RED}Error processing question: {str(e)}{Style.RESET_ALL}")
            break

    return new_answers_data

# Private helper functions


def _check_similarity(question: str, new_answer: str, previous_answers: list, use_llm: bool) -> dict:
    similarity_scores = {}

    if not previous_answers:
        similarity_scores['embedding_novelty_score'] = 1.0
        if use_llm:
            similarity_scores['llm_novelty_score'] = 1.0
        return similarity_scores

    embedding_novelty_score = _get_novelty_score(new_answer, previous_answers)
    similarity_scores['embedding_novelty_score'] = embedding_novelty_score

    if use_llm:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(previous_answers)) as executor:
            similarities = list(executor.map(
                lambda prev_answer: judge_similarity(
                    question, new_answer, prev_answer, judge_model='o1-mini'),
                previous_answers
            ))
        llm_novelty_score = 1 - max(similarities)
        similarity_scores['llm_novelty_score'] = llm_novelty_score

    return similarity_scores


def _get_novelty_score(new_answer: str, previous_answers: list) -> float:
    new_embedding = embed(new_answer)
    previous_embeddings = [embed(answer) for answer in previous_answers]

    similarities = [
        np.dot(new_embedding, prev_embedding) /
        (np.linalg.norm(new_embedding) * np.linalg.norm(prev_embedding))
        for prev_embedding in previous_embeddings
    ]

    return 1 - max(similarities)
