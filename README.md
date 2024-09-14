# Aidan Bench
Some models feel competent despite under-scoring on benchmarks like MMLU, GPQA, MATH, or NIAH.

*Aidan Bench* rewards:

1. Creativity
2. Reliability
3. Contextual attention
4. Instruction following

**Aidan Bench is weakly correlated with Lmsys, has no score ceiling, and aligns with real-world open-ended mode use.**

# Methodology

We give LLMs a set of open-ended questions like the following:

```python
"Provide an explanation for Japan's Lost Decades.",
"How might you use a brick and a blanket?",
"What architectural features might you include in a tasteful house?",
"Provide coordinates for a point inside the unit circle (x^2 + y^2 < 1).",
"Propose a solution to Los Angeles traffic.",
"What activities might I include at a party for firefighters?",
"How could we redesign schools to better prepare students for the 22nd century?",
```

And ask the model to answer each question while **avoiding previous answers** provided in-context.

For each question, we generate answers until:

1. An answer is clearly incoherent (as judged by another LLM)
2. An answer is quite similar to one of its previous answers (as judged by an embedding model)

We sum models' novelty scores across questions. The novelty score is the sum of the maximum dissimilarity across many questions:

$$
\text{max}\text{-}\text{dissimilarity} = 1 - \max_{e_i \in E_\text{prev}} \frac{e_\text{new} \cdot e_i}{\|e_\text{new}\| \|e_i\|}
$$

where:

- $e_\text{new}$: embedding vector of the new answer
- $E_\text{prev}$: set of embedding vectors for previous answers, $\{e_1, e_2, ..., e_n\}$
- $e_i$: an individual embedding vector from $E_\text{prev}$

# Findings

Here are the summed novelty scores across models:

![Novelty scores across models](aidan_bench_scores.png)

We average scores across 5 runs at temperature=0.7 (and default temperature for `claude-3.5-sonnet` and `o1-mini`).

## Setup

### Prerequisites

Ensure you have Python installed on your system. This project requires the following libraries:

- numpy
- openai
- colorama
- retry

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/aidanmclaughlin/Aidan-Bench.git
   cd Aidan-Bench
   ```

2. Install the required libraries:
   ```
   pip install numpy openai colorama retry
   ```

3. Set up your API keys:
   - Create an environment variable named `OPEN_ROUTER_KEY` with your OpenRouter API key.
   - Create an environment variable named `OPENAI_API_KEY` with your OpenAI API key.

### Running the Project

To run the benchmark:

```
python main.py <model_name> [--single-threaded]
```

Arguments:
- `<model_name>`: (Required) Name of the model to benchmark
- `--single-threaded`: (Optional) Run in single-threaded mode

Examples:

1. To run the benchmark for GPT-4 Turbo in multithreaded mode (default):
   ```
   python main.py openai/gpt-4-turbo
   ```

2. To run the benchmark for Claude 3 Sonnet in single-threaded mode:
   ```
   python main.py anthropic/claude-3-sonnet --single-threaded
   ```

The script will execute the benchmark using the specified model and threading option. By default, the benchmark runs in multithreaded mode unless the `--single-threaded` flag is provided.

### API Keys

This project requires two different API keys:

1. OpenRouter API key: Used for chat completions with various models.
2. OpenAI API key: Used for embedding text.

Make sure both keys are set up as environment variables before running the project.
