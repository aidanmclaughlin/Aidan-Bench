import re
from models import chat_with_model
from azure import chat


def gen_answer(question: str, previous_answers: list, model_name: str, cot=False) -> str:
    base_prompt = (
        "Answer the following question:.\n"
        "<question>" + question + "</question>\n"
        "Provide your answer in <answer></answer> XML tags.\n"
    )

    if cot:
        base_prompt += (
            "Let's approach this methodically:\n\n"
            "1. First, carefully review any previous answers to avoid repetition:\n"
            "- Analyze the core concepts and themes already covered\n"
            "- Identify unexplored angles and perspectives\n\n"
            "2. Brainstorm fresh approaches:\n"
            "- Generate multiple unique possibilities\n"
            "- Consider unconventional but valid perspectives\n"
            "- Look for interesting connections or insights\n\n"
            "3. Develop your chosen idea:\n"
            "- Reason through the logic step-by-step\n"
            "- Validate your reasoning\n\n"
            "Show your complete thinking process in <thoughts></thoughts> XML tags.\n"
            "When ready, provide your final response in <answer></answer> XML tags.\n"
        )

    base_prompt += "Your response should be one direct answer- only provide one answer. DO NOT list multiple answers.\n"

    if previous_answers:
        previous_answers_str = "\n".join(
            [f"{i+1}. {answer}" for i, answer in enumerate(previous_answers)]
        )
        base_prompt += (
            "IMPORTANT: Provide an answer you *HAVE NOT* given previously.\n"
            "Your previous answers are inside of <previous_answers></previous_answers> XML tags.\n"
            "<previous_answers>\n" + previous_answers_str + "\n</previous_answers>"
        )

    response = chat_with_model(
        base_prompt, model=model_name, temperature=0.7)
    return _extract_xml_content(response, "answer")


def judge_answer(question: str, answer: str, model_name: str) -> int:
    prompt = (
        "Your task is to evaluate the coherence and plausibility of an answer to a given question.\n\n"
        "Question: <question>" + question + "</question>\n"
        "Answer: <answer>" + answer + "</answer>\n\n"
        "Based on the following criteria, provide a Coherence and Plausibility Score on a scale of 0 - 100:\n\n"
        "0-20: INCOHERENT/NONSENSICAL\n"
        "- Answer is completely unrelated to the question\n"
        "- Contains logical impossibilities or contradictions\n"
        "- Makes claims that defy basic reality\n"
        "- Shows no understanding of the question's context\n\n"
        "21-40: SEVERELY FLAWED\n"
        "- Major logical gaps or inconsistencies\n"
        "- Significant misunderstanding of core concepts\n"
        "- Contains partially relevant information but mostly incorrect\n"
        "- May include some true statements but fails to form a coherent response\n\n"
        "41-60: PARTIALLY COHERENT\n"
        "- Shows basic understanding of the question\n"
        "- Contains some valid points mixed with errors\n"
        "- Logic is followable but may have weak connections\n"
        "- Answer is relevant but may miss key aspects\n\n"
        "61-80: MOSTLY COHERENT\n"
        "- Demonstrates clear understanding of the question\n"
        "- Logic is sound with minor gaps or inconsistencies\n"
        "- Most claims are plausible and well-supported\n"
        "- Forms a generally complete and relevant response\n\n"
        "81-100: HIGHLY COHERENT\n"
        "- Perfectly addresses the question\n"
        "- Demonstrates complete logical consistency\n"
        "- All claims are plausible and well-grounded\n"
        "- Forms a comprehensive and precise response\n\n"
        "IMPORTANT: Provide your final Coherence and Plausibility Score as a single integer between 0 and 100, "
        "enclosed in <coherence_score></coherence_score> XML tags. For example:\n"
        "<coherence_score>75</coherence_score>\n\n"
        "Do not include any additional text in your response."
    )
    response = chat("", prompt, model="o1-mini")
    return int(_extract_xml_content(response, "coherence_score"))


def judge_similarity(question: str, answer1: str, answer2: str, model_name: str) -> float:
    prompt = (
        "Your task is to evaluate how semantically similar two answers are to the same question, "
        "focusing on core concepts and meaning rather than exact wording.\n\n"
        "Original Question: <question>" + question + "</question>\n"
        "First Answer: <answer1>" + answer1 + "</answer1>\n"
        "Second Answer: <answer2>" + answer2 + "</answer2>\n\n"
        "Based on the following criteria, provide a Similarity Score from 0 to 100:\n\n"
        "0-20: Completely different answers with no meaningful overlap\n"
        "21-40: Minimal similarity with few shared concepts\n"
        "41-60: Moderate similarity with some shared core ideas\n"
        "61-80: Substantial similarity with minor variations\n"
        "81-100: Nearly identical in meaning (may use different words)\n\n"
        "IMPORTANT: Provide your final Similarity Score as an integer between 0 and 100, "
        "enclosed in <similarity_score></similarity_score> XML tags. For example:\n"
        "<similarity_score>75</similarity_score>\n\n"
        "Do not include any additional text in your response."
    )
    response = chat("", prompt, model="o1-mini")
    return int(_extract_xml_content(response, "similarity_score")) / 100


def _extract_xml_content(text: str, tag: str) -> str:
    pattern = f"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else text
