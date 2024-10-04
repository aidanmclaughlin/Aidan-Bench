# Questions should be open-ended but demand concrete answers.
questions = [
    "Provide an explanation for Japan's Lost Decades.",
    "What is a cause of World War 1?",
    "Why might the United States government nationalize ASI development?",
    "How might you use a brick and a blanket?",
    "What architectural features might you include in a tasteful house?",
    "Provide coordinates for a point inside the unit circle (x^2 + y^2 < 1).",
    "What's one way to use oregano?",
    "How might we enable LLMs to spend more output tokens to get predictably better results?",
    "Propose a solution to Los Angeles traffic.",
    "What activities might I include at a party for firefighters?",
    "Why did Rome fall?",
    "How could we redesign schools to better prepare students for the 22nd century?",
    "Find a solution to the inequality 2x + 3y < 10 where x and y are positive real numbers.",
    "What might be an unexpected consequence of achieving nuclear fusion?",
    "Describe a plausible alien life form that doesn't rely on carbon-based biology.",
    "How could we modify the rules of chess to make it more exciting for spectators?",
    "What would be the implications of a universal basic income on society?",
    "Propose an alternative to democracy for governing a country.",
    "Provide a real number greater than π but less than 4.",
    "How might we terraform Venus instead of Mars, and why?",
    "Design a new sport that combines elements of three existing sports.",
    "What could be a novel use for blockchain technology outside of cryptocurrency?",
    "How might human evolution be affected by long-term space colonization?",
    "Invent a new musical instrument and describe how it would be played.",
    "What might be an unexpected solution to reducing plastic waste in oceans?",
]


def create_gen_prompt(question: str, previous_answers: list) -> str:
    prompt = (
        "Answer the following question:.\n"
        f"<question>{question}</question>\n"
        "Your response should be a single brief sentence.\n"
    )

    if len(previous_answers) > 0:

        previous_answers_str = "\n".join(
            [f"{i+1}. {answer}" for i, answer in enumerate(previous_answers)]
        )

        prompt += (
            "IMPORTANT: Provide an answer you *HAVE NOT* given previously.\n"
            "Your previous answers are inside of <previous_answers></previous_answers> XML tags.\n"
            f"<previous_answers>\n{previous_answers_str}\n</previous_answers>"
        )

    return prompt


# def create_judge_prompt(question: str, answer: str):
#     prompt = f"""    Your task is to evaluate the coherence and plausibility of an answer to a given question. This involves assessing whether the answer makes sense and isn't nonsensical or implausible.

#     Question: <question>{question}</question>
#     Answer: <answer>{answer}</answer>

#     Evaluation process:
#     1. Understand the question: Analyze what the question is asking.
#     2. Assess the answer: Determine if the answer is coherent and plausible.
#     3. Check for nonsensical elements: Identify any aspects that are completely unrelated or absurd.

#     Please think through each step carefully and show your reasoning:

#     1. Question analysis:
#     [Your brief analysis of the question here]

#     2. Answer assessment:
#     [Evaluate if the answer is coherent and plausible]

#     3. Nonsensical check:
#     [Identify any completely unrelated or absurd elements]

#     Based on your analysis, provide a final Coherence and Plausibility Score on a scale of 1 - 10, where:
#     1-3: Incoherent, implausible, or nonsensical
#     4-6: Partially coherent and plausible, but with some issues
#     7-8: Mostly coherent and plausible with minor issues
#     9-10: Highly coherent and plausible

#     Ensure that nonsensical or completely implausible answers receive very low scores (1-3).

#     IMPORTANT: After your reasoning, you must provide your final Coherence and Plausibility Score as a single integer between 1 and 10, enclosed in <coherence_score></coherence_score> XML tags. For example:
#     <coherence_score>7</coherence_score>

#     Your response must end with this score in the specified format.
#     """
#     return prompt


def create_judge_prompt(question: str, answer: str):
    prompt = f"""
    Your task is to evaluate the quality of an answer to a given question. This involves assessing the answer's accuracy, completeness, relevance, and overall helpfulness.

    Question: <question>{question}</question>
    Answer: <answer>{answer}</answer>

    Evaluation process:
    1. Understand the question: Analyze what the question is asking and identify its key components.
    2. Assess answer accuracy: Determine if the information provided is correct and factual.
    3. Evaluate completeness: Check if the answer addresses all aspects of the question.
    4. Judge relevance: Assess how well the answer relates to the specific question asked.
    5. Consider helpfulness: Determine if the answer provides valuable information or insights to the asker.

    Please think through each step carefully and show your reasoning:

    1. Question analysis:
    [Your brief analysis of the question here]

    2. Accuracy assessment:
    [Evaluate the factual correctness of the answer]

    3. Completeness evaluation:
    [Assess if all aspects of the question are addressed]

    4. Relevance judgment:
    [Determine how well the answer relates to the question]

    5. Helpfulness consideration:
    [Assess the overall value and usefulness of the answer]

    Based on your analysis, provide a final Quality Score on a scale of 1 - 10, where:
    1-3: Poor quality (inaccurate, incomplete, irrelevant, or unhelpful)
    4-6: Moderate quality (partially accurate and helpful, but with significant room for improvement)
    7-8: Good quality (mostly accurate, complete, relevant, and helpful with minor issues)
    9-10: Excellent quality (highly accurate, comprehensive, relevant, and exceptionally helpful)

    Ensure that answers with major inaccuracies or those that fail to address the question receive very low scores (1-3).

    IMPORTANT: After your reasoning, you must provide your final Quality Score as a single integer between 1 and 10, enclosed in <quality_score></quality_score> XML tags. For example:
    <quality_score>8</quality_score>

    Your response must end with this score in the specified format.
    """
    return prompt