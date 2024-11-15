from pathlib import Path
from datetime import datetime
import json
import shutil


def _process_data(data):
    """Convert raw JSON data into the visualization format with aggregate scores"""
    processed_data = {'models': data['models'], 'aggregate_scores': {}}

    for model_name, temps in data['models'].items():
        processed_data['aggregate_scores'][model_name] = {
            'embedding_dissimilarity': 0,
            'llm_dissimilarity': 0,
            'coherence': 0,
            'question_count': 0,
            'answer_count': 0
        }

        for temp_data in temps.values():
            for answers in temp_data.values():
                processed_data['aggregate_scores'][model_name]['question_count'] += 1
                for answer in answers:
                    processed_data['aggregate_scores'][model_name]['embedding_dissimilarity'] += answer['embedding_dissimilarity_score']
                    processed_data['aggregate_scores'][model_name]['coherence'] += answer['coherence_score']
                    if 'llm_dissimilarity_score' in answer:
                        processed_data['aggregate_scores'][model_name]['llm_dissimilarity'] += answer['llm_dissimilarity_score']
                    processed_data['aggregate_scores'][model_name]['answer_count'] += 1

    # Calculate averages
    for model_name, scores in processed_data['aggregate_scores'].items():
        answer_count = scores['answer_count']
        if answer_count > 0:
            scores['embedding_dissimilarity'] = (
                scores['embedding_dissimilarity'] / answer_count) * 100
            scores['llm_dissimilarity'] = (
                scores['llm_dissimilarity'] / answer_count) * 100
            scores['coherence'] = scores['coherence'] / answer_count

    return processed_data


def create_visualization(results_path='results.json'):
    print(f"Loading data from '{results_path}'")

    # Create backup of the JSON file
    backup_dir = Path('backups')
    backup_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = backup_dir / f'results_{timestamp}.json'
    original_json_path = Path(results_path)
    shutil.copy2(original_json_path, backup_path)

    with open(results_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # Add new function to get unique questions
    questions = set()
    for model_data in raw_data['models'].values():
        for temp_data in model_data.values():
            questions.update(temp_data.keys())
    
    data = {
        'raw_data': raw_data,
        'processed_data': _process_data(raw_data),
        'questions': sorted(list(questions))
    }

    viz_dir = Path('visualization')
    viz_dir.mkdir(exist_ok=True)

    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Response Analysis</title>

    <!-- Load dependencies -->
    <script src="https://unpkg.com/react@18.2.0/umd/react.production.min.js"></script>
    <script src="https://unpkg.com/react-dom@18.2.0/umd/react-dom.production.min.js"></script>
    <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <link href="https://unpkg.com/tailwindcss@^2/dist/tailwind.min.css" rel="stylesheet">

    <style>
        .card { 
            background: white;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12);
            margin: 16px;
            padding: 16px;
        }
        .metrics-container {
            display: flex;
            flex-wrap: wrap;
            gap: 16px;
            margin-top: 8px;
        }
        .metric {
            background: #f3f4f6;
            padding: 8px;
            border-radius: 4px;
            flex: 1;
            min-width: 200px;
        }
        .answer {
            margin-top: 16px;
            padding: 16px;
            border: 1px solid #e5e7eb;
            border-radius: 4px;
        }
        .metrics-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
            margin-bottom: 2rem;
        }
        .metrics-table th,
        .metrics-table td {
            padding: 0.5rem;
            border: 1px solid #e5e7eb;
            text-align: center;
        }
        .metrics-table th {
            background-color: #f3f4f6;
        }
        .metrics-table tr:nth-child(even) {
            background-color: #f9fafb;
        }
        .chart-container {
            background: white;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 2rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
    </style>
</head>
<body>
    <div id="root"></div>

    <script type="text/babel">
        const initialData = """ + json.dumps(data) + """;

        function recalculateScores(rawData, excludedQuestions) {
            const filteredData = JSON.parse(JSON.stringify(rawData));
            
            // Remove excluded questions from the data
            Object.entries(filteredData.models).forEach(([modelName, temperatures]) => {
                Object.values(temperatures).forEach(questions => {
                    excludedQuestions.forEach(q => delete questions[q]);
                });
            });
            
            // Recalculate scores using the same logic as _process_data
            const processed = {
                models: filteredData.models,
                aggregate_scores: {}
            };

            Object.entries(filteredData.models).forEach(([modelName, temps]) => {
                processed.aggregate_scores[modelName] = {
                    embedding_dissimilarity: 0,
                    llm_dissimilarity: 0,
                    coherence: 0,
                    question_count: 0,
                    answer_count: 0
                };

                Object.values(temps).forEach(questions => {
                    Object.values(questions).forEach(answers => {
                        processed.aggregate_scores[modelName].question_count += 1;
                        answers.forEach(answer => {
                            processed.aggregate_scores[modelName].embedding_dissimilarity += answer.embedding_dissimilarity_score;
                            processed.aggregate_scores[modelName].coherence += answer.coherence_score;
                            if ('llm_dissimilarity_score' in answer) {
                                processed.aggregate_scores[modelName].llm_dissimilarity += answer.llm_dissimilarity_score;
                            }
                            processed.aggregate_scores[modelName].answer_count += 1;
                        });
                    });
                });

                // Calculate averages
                const scores = processed.aggregate_scores[modelName];
                const answerCount = scores.answer_count;
                if (answerCount > 0) {
                    scores.embedding_dissimilarity = (scores.embedding_dissimilarity / answerCount) * 100;
                    scores.llm_dissimilarity = (scores.llm_dissimilarity / answerCount) * 100;
                    scores.coherence = scores.coherence / answerCount;
                }
            });

            return processed;
        }

        function QuestionFilter({ questions, excludedQuestions, onToggleQuestion }) {
            const handleSelectAll = () => {
                questions.forEach(q => {
                    if (excludedQuestions.includes(q)) {
                        onToggleQuestion(q);
                    }
                });
            };

            const handleSelectNone = () => {
                questions.forEach(q => {
                    if (!excludedQuestions.includes(q)) {
                        onToggleQuestion(q);
                    }
                });
            };

            return (
                <div className="card mb-8">
                    <div className="flex justify-between items-center mb-4">
                        <h2 className="text-xl font-bold">Filter Questions</h2>
                        <div className="space-x-4">
                            <button 
                                onClick={handleSelectAll}
                                className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
                            >
                                Select All
                            </button>
                            <button 
                                onClick={handleSelectNone}
                                className="px-4 py-2 bg-gray-500 text-white rounded hover:bg-gray-600"
                            >
                                Select None
                            </button>
                        </div>
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                        {questions.map(question => (
                            <label key={question} className="flex items-center space-x-2">
                                <input
                                    type="checkbox"
                                    checked={!excludedQuestions.includes(question)}
                                    onChange={() => onToggleQuestion(question)}
                                    className="form-checkbox h-5 w-5 text-blue-600"
                                />
                                <span className="text-sm">{question}</span>
                            </label>
                        ))}
                    </div>
                </div>
            );
        }

        function MetricsChart({ data }) {
            const chartRef = React.useRef(null);
            const timeChartRef = React.useRef(null);

            React.useEffect(() => {
                if (!data || data.length === 0) return;

                // Create metrics plot
                const metricsTraces = [
                    {
                        x: data.map(d => d.iteration),
                        y: data.map(d => d.coherence),
                        name: 'Coherence Score',
                        type: 'scatter',
                        mode: 'lines+markers',
                        line: { color: '#8884d8' }
                    },
                    {
                        x: data.map(d => d.iteration),
                        y: data.map(d => d.embedding),
                        name: 'Embedding Dissimilarity (%)',
                        type: 'scatter',
                        mode: 'lines+markers',
                        line: { color: '#82ca9d' }
                    },
                    {
                        x: data.map(d => d.iteration),
                        y: data.map(d => d.llm),
                        name: 'LLM Dissimilarity (%)',
                        type: 'scatter',
                        mode: 'lines+markers',
                        line: { color: '#ffc658' }
                    }
                ];

                const metricsLayout = {
                    title: 'Metrics Comparison',
                    xaxis: { title: 'Iteration' },
                    yaxis: { title: 'Score' },
                    height: 400,
                    margin: { t: 30 }
                };

                Plotly.newPlot(chartRef.current, metricsTraces, metricsLayout);

                // Create time plot
                const timeTrace = [{
                    x: data.map(d => d.iteration),
                    y: data.map(d => d.time),
                    name: 'Processing Time',
                    type: 'scatter',
                    mode: 'lines+markers',
                    line: { color: '#ff7300' }
                }];

                const timeLayout = {
                    title: 'Processing Time',
                    xaxis: { title: 'Iteration' },
                    yaxis: { title: 'Time (s)' },
                    height: 300,
                    margin: { t: 30 }
                };

                Plotly.newPlot(timeChartRef.current, timeTrace, timeLayout);
            }, [data]);

            if (!data || data.length === 0) return null;

            return (
                <div>
                    <div className="chart-container">
                        <div ref={chartRef}></div>
                    </div>
                    <div className="chart-container">
                        <div ref={timeChartRef}></div>
                    </div>
                </div>
            );
        }

        function MetricsTable({ data }) {
            if (!data || data.length === 0) return null;

            return (
                <div className="overflow-x-auto">
                    <table className="metrics-table">
                        <thead>
                            <tr>
                                <th>Iteration</th>
                                <th>Coherence Score</th>
                                <th>Embedding Dissimilarity</th>
                                <th>Processing Time</th>
                            </tr>
                        </thead>
                        <tbody>
                            {data.map((item, idx) => (
                                <tr key={idx}>
                                    <td>{item.iteration}</td>
                                    <td>{item.coherence}</td>
                                    <td>{item.embedding.toFixed(1)}%</td>
                                    <td>{item.time.toFixed(2)}s</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            );
        }

        function AggregateScores({ scores }) {
            return (
                <div className="card mb-8">
                    <h2 className="text-xl font-bold mb-4">Aggregate Model Scores</h2>
                    <div className="overflow-x-auto">
                        <table className="metrics-table">
                            <thead>
                                <tr>
                                    <th>Model</th>
                                    <th>Avg Coherence</th>
                                    <th>Avg Embedding Dissimilarity</th>
                                    <th>Avg LLM Dissimilarity</th>
                                    <th>Questions</th>
                                    <th>Total Responses</th>
                                </tr>
                            </thead>
                            <tbody>
                                {Object.entries(scores)
                                    .sort(([, a], [, b]) => b.answer_count - a.answer_count)
                                    .map(([model, stats]) => (
                                        <tr key={model}>
                                            <td>{model}</td>
                                            <td>{stats.coherence.toFixed(3)}</td>
                                            <td>{stats.embedding_dissimilarity.toFixed(1)}%</td>
                                            <td>{stats.llm_dissimilarity.toFixed(1)}%</td>
                                            <td>{stats.question_count}</td>
                                            <td>{stats.answer_count}</td>
                                        </tr>
                                    ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            );
        }

        function ModelResponses() {
            const [excludedQuestions, setExcludedQuestions] = React.useState([]);
            const [processedData, setProcessedData] = React.useState(initialData.processed_data);
            const [expandedModel, setExpandedModel] = React.useState(null);
            const [expandedTemperature, setExpandedTemperature] = React.useState(null);
            const [expandedQuestion, setExpandedQuestion] = React.useState(null);

            const handleToggleQuestion = (question) => {
                const newExcluded = excludedQuestions.includes(question)
                    ? excludedQuestions.filter(q => q !== question)
                    : [...excludedQuestions, question];
                setExcludedQuestions(newExcluded);
                
                const newProcessedData = recalculateScores(initialData.raw_data, newExcluded);
                setProcessedData(newProcessedData);
            };

            const prepareMetricsData = (answers) => {
                return answers.map(answer => ({
                    iteration: answer.answer_num,
                    coherence: answer.coherence_score,
                    embedding: answer.embedding_dissimilarity_score * 100,
                    llm: (answer.llm_dissimilarity_score || 0) * 100,
                    time: answer.processing_time
                }));
            };

            return (
                <div className="container mx-auto p-4 max-w-6xl">
                    <h1 className="text-2xl font-bold mb-4">Model Response Analysis</h1>
                    
                    <QuestionFilter
                        questions={initialData.questions}
                        excludedQuestions={excludedQuestions}
                        onToggleQuestion={handleToggleQuestion}
                    />
                    
                    <AggregateScores scores={processedData.aggregate_scores} />

                    {Object.entries(processedData.models).map(([modelName, temperatures]) => (
                        <div key={modelName} className="card">
                            <button
                                className="w-full text-left font-bold text-lg p-2 hover:bg-gray-50 rounded flex items-center justify-between"
                                onClick={() => setExpandedModel(expandedModel === modelName ? null : modelName)}
                            >
                                <span>{modelName}</span>
                                <span>{expandedModel === modelName ? '▼' : '▶'}</span>
                            </button>

                            {expandedModel === modelName && Object.entries(temperatures).map(([temp, questions]) => (
                                <div key={temp} className="ml-4 mt-4 border-l-2 border-gray-200 pl-4">
                                    <button
                                        className="w-full text-left font-medium p-2 hover:bg-gray-50 rounded flex items-center justify-between"
                                        onClick={() => setExpandedTemperature(expandedTemperature === temp ? null : temp)}
                                    >
                                        <span>Temperature: {temp}</span>
                                        <span>{expandedTemperature === temp ? '▼' : '▶'}</span>
                                    </button>

                                    {expandedTemperature === temp && Object.entries(questions).map(([question, answers]) => (
                                        <div key={question} className="ml-4 mt-4 border-l-2 border-gray-200 pl-4">
                                            <button
                                                className="w-full text-left font-medium p-2 hover:bg-gray-50 rounded flex items-center justify-between"
                                                onClick={() => setExpandedQuestion(expandedQuestion === question ? null : question)}
                                            >
                                                <span>{question}</span>
                                                <span>{expandedQuestion === question ? '▼' : '▶'}</span>
                                            </button>

                                            {expandedQuestion === question && (
                                                <div className="ml-4 mt-4">
                                                    <MetricsChart data={prepareMetricsData(answers)} />
                                                    <MetricsTable data={prepareMetricsData(answers)} />

                                                    <div className="space-y-6">
                                                        {answers.map((answer, idx) => (
                                                            <div key={idx} className="answer bg-white">
                                                                <div className="font-medium text-lg mb-2">Response {answer.answer_num}</div>
                                                                <div className="metrics-container mb-4">
                                                                    <div className="metric">
                                                                        <span className="font-medium">Coherence:</span> {answer.coherence_score}
                                                                    </div>
                                                                    <div className="metric">
                                                                        <span className="font-medium">Embedding Dissimilarity:</span> {(answer.embedding_dissimilarity_score * 100).toFixed(1)}%
                                                                    </div>
                                                                    {answer.llm_dissimilarity_score !== undefined && (
                                                                        <div className="metric">
                                                                            <span className="font-medium">LLM Dissimilarity:</span> {(answer.llm_dissimilarity_score * 100).toFixed(1)}%
                                                                        </div>
                                                                    )}
                                                                    <div className="metric">
                                                                        <span className="font-medium">Processing Time:</span> {answer.processing_time.toFixed(2)}s
                                                                    </div>
                                                                </div>
                                                                <div className="mt-2 p-4 bg-gray-50 rounded whitespace-pre-wrap">
                                                                    {answer.answer}
                                                                </div>
                                                            </div>
                                                        ))}
                                                    </div>
                                                </div>
                                            )}
                                        </div>
                                    ))}
                                </div>
                            ))}
                        </div>
                    ))}
                </div>
            );
        }

        const root = ReactDOM.createRoot(document.getElementById('root'));
        root.render(<ModelResponses />);
    </script>
</body>
</html>
    """

    output_file = viz_dir / 'index.html'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content.replace("</script>", "</" + "script>"))

    print(f"\nVisualization created successfully!")
    print(f"To view the visualization:")
    print(f"1. Run: python -m http.server 8000")
    print(f"2. Open: http://localhost:8000/visualization")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Create visualization from results JSON file')
    parser.add_argument('results_path', help='Path to the results JSON file')
    args = parser.parse_args()

    create_visualization(args.results_path)

# Private helper functions below
