extended_eval = True

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

extended_questions = [
    # Creative and unexpected questions - generated from the base.
    "How might we design a city that functions entirely underwater?",
    "What societal changes might occur if humans could communicate with animals?",
    "Propose a non-military use for a fleet of drones.",
    "Describe a sustainable farming method that could be used in a floating city.",
    "If all buildings were required to be bio-luminescent, what effects might this have?",
    "Invent a device that translates dreams into tangible art.",
    "How might daily life change if humans had the ability to breathe underwater?",
    "Create a recipe for a 'happiness-inducing' smoothie.",
    "What new environmental challenges might arise if all vehicles were self-driving?",
    "Design a fashion line that incorporates smart clothing technology.",
    "Imagine a world where books are replaced by holographic storytelling; what impacts might this have?",
    "What might be the implications of having robots as therapists?",
    "Propose a system for energy-harvesting from natural disasters.",
    "How might the education system be revolutionized by virtual reality classrooms?",
    "What unique challenges might arise in a society where everyone lives to be 150 years old?",
    "Describe a mobile app that encourages acts of kindness.",
    "If people could naturally hibernate, what would be the societal and economic impacts?",
    "Imagine a competition where contestants build habitats for animals; what might be included?",
    "What might be the benefits of reintroducing dinosaurs into modern ecosystems?",
    "Propose a mechanism for reducing food waste through technological innovation.",
    "Design a city where all modes of transportation are vertically oriented.",
    "What societal norms might emerge in a world without gender?",
    "Imagine a civilization based entirely on underwater agriculture; what technology might be required?",
    "How might public health improve if all houses had healing gardens?",
    "What might be the cultural effects if everyone suddenly had access to time travel?",
    "Create a concept for a museum that showcases possible futures.",
    "What would be the impact of a government mandating weekly mental health days?",
    "Invent a game that teaches players about sustainable living.",
    "How could we design a school that encourages lifelong learning from adults as well as children?",
    "Describe a new form of professional sports that focuses on non-physical competition.",
]

# if you want an extended evaluation
if extended_eval:
    questions += extended_questions


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
