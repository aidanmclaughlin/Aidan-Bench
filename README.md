# Novel-Gen-Bench
Some models feel competent despite under-scoring on benchmarks like MMLU, GPQA, MATH, or NIAH.

*Aidan Bench* rewards models that:

1. Are creative
2. Are reliable
3. Attend to context
4. Follow instructions

Aidan Bench's champions closely (but inexactly) match my intuition for the world's best LLMs. Topology uses Aidan Bench to gauge the quality of products like the CLM.

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

We sum models' novelty score across questions. The novelty score is the sum of the maximum dissimilarity across many questions:

$$\text{max\_dissimilarity} = 1 - \max_{e_i \in E_\text{prev}} \frac{e_\text{new} \cdot e_i}{\|e_\text{new}\| \|e_i\|}$$

where:

- $e_\text{new}$: embedding vector of the new answer
- $E_\text{prev}$: set of embedding vectors for previous answers, $\{e_1, e_2, ..., e_n\}$
- $e_i$: an individual embedding vector from $E_\text{prev}$

# Findings

Here are the final novelty scores across models:

![Novelty scores across models](output-4.png)

Notable results:

1. `Mistral Large 2` wins this benchmark, scoring 25% higher than `Claude 3.5 Sonnet`, the runner-up.
2. OpenAI's `GPT-4o` underperforms similarly priced models substantially, including its cheaper sibling, `GPT-4o-mini`.
3. OpenAI's `GPT-4o-mini` punches well above its price class, rivaling much more expensive models like `Llama 3.1 405b`.

We also include a comparison between Aidan Bench scores and Lmsys scores. Notably, there's a weak correlation between these benchmarks (r=0.188).

![Comparison of Aidan Bench and Lmsys scores](output-5.png)

We also compare each model's Aidan Bench scores to its (input) token pricing:

![Comparison of Aidan Bench scores and token pricing](output-7.png)

OpenAI's `GPT-4o-mini` and `Mistral Large 2` have outlier efficiency.
