"""A simple script to test how a model responds to a question posed in AidanBench format."""
from prompts import gen_answer, judge_answer, judge_similarity

question_to_test = "What might be an unexpected consequence of achieving nuclear fusion?"

answers = []
for _ in range(10):
    answer = gen_answer(question_to_test, answers, model_name="openai/gpt-4o")
    answers.append(answer)
    print("="*30)
    print(answer)
