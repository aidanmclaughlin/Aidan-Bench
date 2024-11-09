
questions = [
    (1, "Provide an explanation for Japan's Lost Decades."),
    (2, "What is a cause of World War 1?"),
    (3, "Why might the United States government nationalize ASI development?"),
    (4, "How might you use a brick and a blanket?"),
    (5, "What architectural features might you include in a tasteful house?"),
    (6, "Provide coordinates for a point inside the unit circle (x^2 + y^2 < 1)."),
    (7, "What's one way to use oregano?"),
    (8, "How might we enable LLMs to spend more output tokens to get predictably better results?"),
    (9, "Propose a solution to Los Angeles traffic."),
    (10, "What activities might I include at a party for firefighters?"),
    (11, "Why did Rome fall?"),
    (12, "How could we redesign schools to better prepare students for the 22nd century?"),
    (13, "Find a solution to the inequality 2x + 3y < 10 where x and y are positive real numbers."),
    (14, "What might be an unexpected consequence of achieving nuclear fusion?"),
    (15, "Describe a plausible alien life form that doesn't rely on carbon-based biology."),
    (16, "How could we modify the rules of chess to make it more exciting for spectators?"),
    (17, "What would be the implications of a universal basic income on society?"),
    (18, "Propose an alternative to democracy for governing a country."),
    (19, "Provide a real number greater than π but less than 4."),
    (20, "How might we terraform Venus instead of Mars, and why?"),
    (21, "Design a new sport that combines elements of three existing sports."),
    (22, "What could be a novel use for blockchain technology outside of cryptocurrency?"),
    (23, "How might human evolution be affected by long-term space colonization?"),
    (24, "Invent a new musical instrument and describe how it would be played."),
    (25, "What might be an unexpected solution to reducing plastic waste in oceans?"),
]

extended_questions = [
    (26, "How might we design a city that functions entirely underwater?"),
    (27, "What societal changes might occur if humans could communicate with animals?"),
    (28, "Propose a non-military use for a fleet of drones."),
    (29, "Describe a sustainable farming method that could be used in a floating city."),
    (30, "If all buildings were required to be bio-luminescent, what effects might this have?"),
    (31, "Invent a device that translates dreams into tangible art."),
    (32, "How might daily life change if humans had the ability to breathe underwater?"),
    (33, "Create a recipe for a 'happiness-inducing' smoothie."),
    (34, "What new environmental challenges might arise if all vehicles were self-driving?"),
    (35, "Design a fashion line that incorporates smart clothing technology."),
    (36, "Imagine a world where books are replaced by holographic storytelling; what impacts might this have?"),
    (37, "What might be the implications of having robots as therapists?"),
    (38, "Propose a system for energy-harvesting from natural disasters."),
    (39, "How might the education system be revolutionized by virtual reality classrooms?"),
    (40, "What unique challenges might arise in a society where everyone lives to be 150 years old?"),
    (41, "Describe a mobile app that encourages acts of kindness."),
    (42, "If people could naturally hibernate, what would be the societal and economic impacts?"),
    (43, "Imagine a competition where contestants build habitats for animals; what might be included?"),
    (44, "What might be the benefits of reintroducing dinosaurs into modern ecosystems?"),
    (45, "Propose a mechanism for reducing food waste through technological innovation."),
    (46, "Design a city where all modes of transportation are vertically oriented."),
    (47, "What societal norms might emerge in a world without gender?"),
    (48, "Imagine a civilization based entirely on underwater agriculture; what technology might be required?"),
    (49, "How might public health improve if all houses had healing gardens?"),
    (50, "What might be the cultural effects if everyone suddenly had access to time travel?"),
    (51, "Create a concept for a museum that showcases possible futures."),
    (52, "What would be the impact of a government mandating weekly mental health days?"),
    (53, "Invent a game that teaches players about sustainable living."),
    (54, "How could we design a school that encourages lifelong learning from adults as well as children?"),
    (55, "Describe a new form of professional sports that focuses on non-physical competition."),
]

extended_eval = False

if extended_eval:
    questions += extended_questions


def create_gen_prompt(question: str, previous_answers: list, cot=False) -> str:
    if cot:
        prompt = (
            "Answer the following question:.\n"
            f"<question>{question}</question>\n"
            "Please think carefully step-by-step and show your reasoning.\n"
            "Enclose your thoughts in <thoughts></thoughts> XML tags.\n"
            "Finally, when you are ready to give your response, enclose it in <answer></answer> XML tags.\n"
            "Your final answer should be a single brief sentence.\n"
        )
    
    else:
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


def create_judge_prompt(question: str, answer: str):
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
    return prompt


# def create_judge_prompt(question: str, answer: str):
#     prompt = f"""
#     Your task is to evaluate the quality of an answer to a given question. This involves assessing the answer's accuracy, completeness, relevance, and overall helpfulness.

#     Question: <question>{question}</question>
#     Answer: <answer>{answer}</answer>

#     Evaluation process:
#     1. Understand the question: Analyze what the question is asking and identify its key components.
#     2. Assess answer accuracy: Determine if the information provided is correct and factual.
#     3. Evaluate completeness: Check if the answer addresses all aspects of the question.
#     4. Judge relevance: Assess how well the answer relates to the specific question asked.
#     5. Consider helpfulness: Determine if the answer provides valuable information or insights to the asker.

#     Please think through each step carefully and show your reasoning:

#     1. Question analysis:
#     [Your brief analysis of the question here]

#     2. Accuracy assessment:
#     [Evaluate the factual correctness of the answer]

#     3. Completeness evaluation:
#     [Assess if all aspects of the question are addressed]

#     4. Relevance judgment:
#     [Determine how well the answer relates to the question]

#     5. Helpfulness consideration:
#     [Assess the overall value and usefulness of the answer]

#     Based on your analysis, provide a final Quality Score on a scale of 1 - 10, where:
#     1-3: Poor quality (inaccurate, incomplete, irrelevant, or unhelpful)
#     4-6: Moderate quality (partially accurate and helpful, but with significant room for improvement)
#     7-8: Good quality (mostly accurate, complete, relevant, and helpful with minor issues)
#     9-10: Excellent quality (highly accurate, comprehensive, relevant, and exceptionally helpful)

#     Ensure that answers with major inaccuracies or those that fail to address the question receive very low scores (1-3).

#     IMPORTANT: After your reasoning, you must provide your final Quality Score as a single integer between 1 and 10, enclosed in <quality_score></quality_score> XML tags. For example:
#     <quality_score>8</quality_score>

#     Your response must end with this score in the specified format.
#     """
#     return prompt