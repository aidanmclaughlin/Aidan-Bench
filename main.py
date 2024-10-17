import numpy as np
from models import chat_with_model, embed
from prompts import questions, create_gen_prompt, create_judge_prompt
from colorama import Fore, Style
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import argparse
import statistics
import matplotlib.pyplot as plt
import csv
from collections import defaultdict
import traceback
import json  # Added import for JSON handling


def parse_arguments():
    parser = argparse.ArgumentParser(description="Benchmark a language model.")
    parser.add_argument("model_name", type=str,
                        help="Name of the model to benchmark")
    parser.add_argument("--single-threaded", action="store_true",
                        help="Run in single-threaded mode")
    return parser.parse_args()


def benchmark_model(model_name, multithreaded=False, temperature=0.7):
    if multithreaded:
        return benchmark_model_multithreaded(model_name, temperature)
    else:
        return benchmark_model_sequential(model_name, temperature)


def process_question(question, model_name, temperature):
    start_time = time.time()
    print(Fore.RED + question + Style.RESET_ALL)
    previous_answers = []
    question_novelty = 0
    question_data = []  # List to store data per answer

    try:
        while True:
            gen_prompt = create_gen_prompt(question, previous_answers)
            try:
                new_answer = chat_with_model(
                    prompt=gen_prompt, model=model_name, temperature=temperature)
            except Exception as e:
                print(
                    Fore.RED + f"Error generating answer: {str(e)}" + Style.RESET_ALL)
                break

            judge_prompt = create_judge_prompt(question, new_answer)
            judge = "gpt-4o-mini"
            try:
                judge_response = chat_with_model(
                    prompt=judge_prompt, model=judge)
            except Exception as e:
                print(
                    Fore.RED + f"Error getting judge response: {str(e)}" + Style.RESET_ALL)
                break

            coherence_score = int(judge_response.split("<coherence_score>")[1].split("</coherence_score>")[0])

            if coherence_score <= 3:
                print(
                    Fore.YELLOW + "Output is incoherent. Moving to next question." + Style.RESET_ALL)
                break

            novelty_score = get_novelty_score(new_answer, previous_answers)

            if novelty_score < 0.1:
                print(
                    Fore.YELLOW + "Output is redundant. Moving to next question." + Style.RESET_ALL)
                break

            print(f"Question:\n{question}")
            print(f"New Answer:\n{new_answer}")
            print(Fore.GREEN + f"Coherence Score: {coherence_score}")
            print(f"Novelty Score: {novelty_score}" + Style.RESET_ALL)

            previous_answers.append(new_answer)
            question_novelty += novelty_score

            # Collect data for each answer
            question_data.append({
                'question': question,
                'answer': new_answer,
                'coherence_score': coherence_score,
                'novelty_score': novelty_score
            })

    except Exception as e:
        print(
            Fore.RED + f"Unexpected error processing question: {str(e)}" + Style.RESET_ALL)
        print(Fore.RED + traceback.format_exc() + Style.RESET_ALL)

    time_taken = time.time() - start_time
    print(Fore.BLUE)
    print(f"Total novelty score for this question: {question_novelty}")
    print(f"Time taken: {time_taken} seconds")
    print(Style.RESET_ALL)

    return question_novelty, question_data  # Return collected data


def get_novelty_score(new_answer: str, previous_answers: list):
    new_embedding = embed(new_answer)

    # If there are no previous answers, return maximum novelty
    if not previous_answers:
        return 1.0

    previous_embeddings = [embed(answer) for answer in previous_answers]

    similarities = [
        np.dot(new_embedding, prev_embedding) /
        (np.linalg.norm(new_embedding) * np.linalg.norm(prev_embedding))
        for prev_embedding in previous_embeddings
    ]

    max_similarity = max(similarities)
    novelty = 1 - max_similarity

    return novelty


def benchmark_model_multithreaded(model_name, temperature):
    novelty_score = 0
    print_lock = threading.Lock()
    all_question_data = []  # List to store data from all questions

    with ThreadPoolExecutor(max_workers=len(questions)) as executor:
        future_to_question = {executor.submit(
            process_question, question, model_name, temperature): question for question in questions}

        for future in as_completed(future_to_question):
            question = future_to_question[future]

            try:
                question_novelty, question_data = future.result()
                with print_lock:
                    novelty_score += question_novelty
                    all_question_data.extend(question_data)
            except Exception as e:
                print(
                    Fore.RED + f"Error processing question '{question}': {str(e)}" + Style.RESET_ALL)
                print(Fore.RED + traceback.format_exc() + Style.RESET_ALL)

    print(Fore.YELLOW)
    print(f"Total novelty score across all questions: {novelty_score}")
    print(Style.RESET_ALL)

    # Save all collected data to a JSON file
    filename = f'benchmark_{model_name.replace("/", "-")}_{int(time.time())}.json'
    with open(filename, 'w') as f:
        json.dump(all_question_data, f, indent=2)
    print(f"{Fore.YELLOW}Saved detailed results to {filename}{Style.RESET_ALL}")

    return novelty_score


def benchmark_model_sequential(model_name, temperature):
    novelty_score = 0
    all_question_data = []  # List to store data from all questions

    for question in questions:
        question_novelty, question_data = process_question(
            question, model_name, temperature)
        novelty_score += question_novelty
        all_question_data.extend(question_data)

    print(Fore.YELLOW)
    print(f"Total novelty score across all questions: {novelty_score}")
    print(Style.RESET_ALL)

    # Save all collected data to a JSON file
    filename = f'benchmark_{model_name.replace("/", "-")}_{int(time.time())}.json'
    with open(filename, 'w') as f:
        json.dump(all_question_data, f, indent=2)
    print(f"{Fore.YELLOW}Saved detailed results to {filename}{Style.RESET_ALL}")

    return novelty_score



def read_existing_csv(filename):
    try:
        with open(filename, "r", newline="") as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)
            return {row[0]: [cell.strip() for cell in row[1:]] for row in reader}
    except FileNotFoundError:
        return {}



if __name__ == "__main__":
    args = parse_arguments()
    benchmark_model(
        args.model_name, multithreaded=not args.single_threaded, temperature=0.7)