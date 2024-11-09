import re
from models import chat_with_model


def gen_answer(question: str, previous_answers: list, cot=False) -> str:
    base_prompt = (
        "Answer the following question:.\n"
        f"<question>{question}</question>\n"
    )

    if cot:
        base_prompt += (
            "Please think carefully step-by-step and show your reasoning.\n"
            "Enclose your thoughts in <thoughts></thoughts> XML tags.\n"
            "Finally, when you are ready to give your response, enclose it in <answer></answer> XML tags.\n"
        )

    base_prompt += "Your response should be a single brief sentence.\n"

    if previous_answers:
        previous_answers_str = "\n".join(
            [f"{i+1}. {answer}" for i, answer in enumerate(previous_answers)]
        )
        base_prompt += (
            "IMPORTANT: Provide an answer you *HAVE NOT* given previously.\n"
            "Your previous answers are inside of <previous_answers></previous_answers> XML tags.\n"
            f"<previous_answers>\n{previous_answers_str}\n</previous_answers>"
        )

    response = chat_with_model(
        base_prompt, temperature=0.7, chain_of_thought=cot)
    return _extract_xml_content(response, "answer")


def judge_answer(question: str, answer: str) -> int:
    prompt = f"""    Your task is to evaluate the coherence and plausibility of an answer to a given question. This involves assessing whether the answer makes sense and isn't nonsensical or implausible.

    Question: <question>{question}</question>
    Answer: <answer>{answer}</answer>

    Evaluation process:
    1. Understand the question: Analyze what the question is asking.
    2. Assess the answer: Determine if the answer is coherent and plausible.
    3. Check for nonsensical elements: Identify any aspects that are completely unrelated or absurd.

    Please think through each step carefully and show your reasoning:

    1. Question analysis:
    [Your brief analysis of the question here]

    2. Answer assessment:
    [Evaluate if the answer is coherent and plausible]

    3. Nonsensical check:
    [Identify any completely unrelated or absurd elements]

    Based on your analysis, provide a final Coherence and Plausibility Score on a scale of 1 - 10, where:
    1-3: Incoherent, implausible, or nonsensical
    4-6: Partially coherent and plausible, but with some issues
    7-8: Mostly coherent and plausible with minor issues
    9-10: Highly coherent and plausible

    Ensure that nonsensical or completely implausible answers receive very low scores (1-3).

    IMPORTANT: After your reasoning, you must provide your final Coherence and Plausibility Score as a single integer between 1 and 10, enclosed in <coherence_score></coherence_score> XML tags. For example:
    <coherence_score>7</coherence_score>

    Your response must end with this score in the specified format.
    """
    response = chat_with_model(prompt, temperature=0.7)
    return int(_extract_xml_content(response, "coherence_score"))


def judge_similarity(question: str, answer1: str, answer2: str) -> float:
    prompt = f"""Your task is to evaluate how semantically similar two answers are to the same question, focusing on core concepts and meaning rather than exact wording.

    Original Question: <question>{question}</question>
    First Answer: <answer1>{answer1}</answer1>
    Second Answer: <answer2>{answer2}</answer2>

    Evaluation process:
    1. Core Concept Analysis:
    - Identify the main ideas and concepts in each answer
    - List key elements that overlap or differ

    2. Semantic Comparison:
    - Compare the underlying meaning of each answer
    - Note any significant differences in approach or perspective
    - Consider if the answers could be considered paraphrases of each other

    3. Context Consideration:
    - Evaluate how each answer relates to the original question
    - Determine if differences are substantive or merely superficial

    Based on your analysis, provide a Similarity Score from 0 to 100, where:
    0-20: Completely different answers with no meaningful overlap
    21-40: Minimal similarity with few shared concepts
    41-60: Moderate similarity with some shared core ideas
    61-80: Substantial similarity with minor variations
    81-100: Nearly identical in meaning (may use different words)

    IMPORTANT: After your reasoning, provide your final Similarity Score as an integer between 0 and 100, enclosed in <similarity_score></similarity_score> XML tags. For example:
    <similarity_score>75</similarity_score>

    Your response must end with this score in the specified format.
    """
    response = chat_with_model(prompt, temperature=0.7)
    return int(_extract_xml_content(response, "similarity_score")) / 100


def _extract_xml_content(text: str, tag: str) -> str:
    pattern = f"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else text
