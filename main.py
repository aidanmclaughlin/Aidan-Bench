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

            coherence_score = int(judge_response.split("<coherence_score>")[
                1].split("</coherence_score>")[0])

            if coherence_score <= 3:
                print(
                    Fore.YELLOW + "Output is incoherent. Moving to next question." + Style.RESET_ALL)
                break

            novelty_score = get_novelty_score(new_answer, previous_answers)

            if novelty_score < 0.1:
                print(
                    Fore.YELLOW + "Output is redundant. Moving to next question." + Style.RESET_ALL)
                break

            print(f"New Answer:\n{new_answer}")
            print(Fore.GREEN + f"Coherence Score: {coherence_score}")
            print(f"Novelty Score: {novelty_score}" + Style.RESET_ALL)

            previous_answers.append(new_answer)
            question_novelty += novelty_score

    except Exception as e:
        print(
            Fore.RED + f"Unexpected error processing question: {str(e)}" + Style.RESET_ALL)
        print(Fore.RED + traceback.format_exc() + Style.RESET_ALL)

    time_taken = time.time() - start_time
    print(Fore.BLUE)
    print(f"Total novelty score for this question: {question_novelty}")
    print(f"Time taken: {time_taken} seconds")
    print(Style.RESET_ALL)

    return question_novelty


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

    with ThreadPoolExecutor(max_workers=len(questions)) as executor:
        future_to_question = {executor.submit(
            process_question, question, model_name, temperature): question for question in questions}

        for future in as_completed(future_to_question):
            question = future_to_question[future]

            question_novelty = future.result()
            with print_lock:
                novelty_score += question_novelty

    print(Fore.YELLOW)
    print(f"Total novelty score across all questions: {novelty_score}")
    print(Style.RESET_ALL)

    return novelty_score


def benchmark_model_sequential(model_name, temperature):
    novelty_score = 0

    for question in questions:
        question_novelty = process_question(question, model_name, temperature)
        novelty_score += question_novelty

    print(Fore.YELLOW)
    print(f"Total novelty score across all questions: {novelty_score}")
    print(Style.RESET_ALL)

    return novelty_score


def run_multiple_benchmarks(model_name, multithreaded=False):
    temperatures = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    csv_filename = "benchmark_results.csv"

    # Read existing CSV data
    existing_data = read_existing_csv(csv_filename)

    # Determine missing temperatures for the model
    missing_temps = get_missing_temperatures(existing_data, model_name, temperatures)

    # If all temperatures are present, print a message and return
    if not missing_temps:
        print(f"{Fore.YELLOW}All temperatures already benchmarked for {model_name}. No new runs needed.{Style.RESET_ALL}")
        return

    # Create CSV file if it doesn't exist and write header
    if not existing_data:
        with open(csv_filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            header = ["Model"] + [f"Temp {temp}" for temp in temperatures]
            writer.writerow(header)

    # Load existing results for the model
    results = existing_data.get(model_name, [None] * len(temperatures))

    for temp in missing_temps:
        print(f"\n{Fore.CYAN}Running benchmark with temperature {temp}{Style.RESET_ALL}")
        try:
            score = benchmark_model(model_name, multithreaded, temperature=temp)
            print(f"{Fore.GREEN}Run completed with temperature {temp} and score: {score}{Style.RESET_ALL}")
            results[temperatures.index(temp)] = score
        except Exception as e:
            print(f"{Fore.RED}Error in run with temperature {temp}: {str(e)}{Style.RESET_ALL}")
            results[temperatures.index(temp)] = f"ERROR: {str(e)}"

        # Update CSV file after each temperature run
        update_csv_file(csv_filename, model_name, results)

    print(f"{Fore.YELLOW}All runs completed. Results saved to {csv_filename}{Style.RESET_ALL}")


def read_existing_csv(filename):
    try:
        with open(filename, "r", newline="") as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)
            return {row[0]: [cell.strip() for cell in row[1:]] for row in reader}
    except FileNotFoundError:
        return {}


def get_missing_temperatures(existing_data, model_name, temperatures):
    if model_name not in existing_data:
        return temperatures
    existing_temps = existing_data[model_name]
    missing_temps = []
    for i, temp in enumerate(temperatures):
        if i >= len(existing_temps) or not existing_temps[i] or existing_temps[i] == "None":
            missing_temps.append(temp)
    return missing_temps


def update_csv_file(filename, model_name, scores):
    # Read existing data
    existing_data = defaultdict(list)
    with open(filename, "r", newline="") as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        for row in reader:
            existing_data[row[0]] = row[1:]

    # Update data for the current model
    existing_data[model_name] = scores

    # Write updated data back to CSV
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for model, model_scores in existing_data.items():
            writer.writerow([model] + model_scores)


if __name__ == "__main__":
    args = parse_arguments()
    run_multiple_benchmarks(
        args.model_name, multithreaded=not args.single_threaded)
