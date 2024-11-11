# AidanBench Results Visualizer

This tool creates an interactive visualization for analyzing model responses and their associated metrics. It takes a pickle file containing model response data and creates a web-based dashboard.

## Prerequisites

- Running main.py will give results in a pkl file in the following format:

```python
{
    'models': {
        'model_name': {
            temperature: {
                'question': [
                    {
                        'answer_num': int,
                        'answer': str,
                        'embedding_dissimilarity_score': float,
                        'coherence_score': float,
                        'processing_time': float,
                        'llm_dissimilarity_score': float
                    },
                    # ... more answers
                ]
            }
        }
    }
}
```

## Usage

1. Run the visualization script with your pickle file:
```bash
python visualize.py path/to/your/data.pkl
```

2. Start a local server:
```bash
python -m http.server 8000
```

3. Open your web browser and navigate to:
```
http://localhost:8000/visualization
```

## Features

The visualization includes:
- Expandable sections for each model and question
- Interactive charts showing:
  - Coherence scores, embedding dissimilarity, and LLM dissimilarity across iterations
  - Processing time trends
- Detailed metrics table
- Full response text for each iteration
