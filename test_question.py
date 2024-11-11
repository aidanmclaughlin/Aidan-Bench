"""A simple script to test how a model responds to a question posed in AidanBench format."""
from prompts import gen_answer, judge_answer, judge_similarity

question_to_test = "Imagine a world where books are replaced by holographic storytelling; what impacts might this have?"

answers = []
for _ in range(10):
    answer = gen_answer(question_to_test, answers, model_name="openai/gpt-4o")
    answers.append(answer)
    judge_answer(question_to_test, answer, model_name="openai/gpt-4o")
    judge_similarity(question_to_test, answer, answers[0], model_name="openai/gpt-4o")

    print("="*30)
    print(answer)
