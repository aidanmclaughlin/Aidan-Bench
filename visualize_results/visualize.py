import pickle
import json
import sys
import os
from pathlib import Path

def create_visualization(pkl_path):
    print(f"Loading data from: {pkl_path}")
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    viz_dir = Path('visualization')
    viz_dir.mkdir(exist_ok=True)
    
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AidanBench Visualizer</title>
    
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
        const modelData = """ + json.dumps(data) + """;
        
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
                        line: {color: '#8884d8'}
                    },
                    {
                        x: data.map(d => d.iteration),
                        y: data.map(d => d.embedding),
                        name: 'Embedding Dissimilarity (%)',
                        type: 'scatter',
                        mode: 'lines+markers',
                        line: {color: '#82ca9d'}
                    },
                    {
                        x: data.map(d => d.iteration),
                        y: data.map(d => d.llm),
                        name: 'LLM Dissimilarity (%)',
                        type: 'scatter',
                        mode: 'lines+markers',
                        line: {color: '#ffc658'}
                    }
                ];

                const metricsLayout = {
                    title: 'Metrics Comparison',
                    xaxis: {title: 'Iteration'},
                    yaxis: {title: 'Score'},
                    height: 400,
                    margin: {t: 30}
                };

                Plotly.newPlot(chartRef.current, metricsTraces, metricsLayout);

                // Create time plot
                const timeTrace = [{
                    x: data.map(d => d.iteration),
                    y: data.map(d => d.time),
                    name: 'Processing Time',
                    type: 'scatter',
                    mode: 'lines+markers',
                    line: {color: '#ff7300'}
                }];

                const timeLayout = {
                    title: 'Processing Time',
                    xaxis: {title: 'Iteration'},
                    yaxis: {title: 'Time (s)'},
                    height: 300,
                    margin: {t: 30}
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
                                <th>LLM Dissimilarity</th>
                                <th>Processing Time</th>
                            </tr>
                        </thead>
                        <tbody>
                            {data.map((item, idx) => (
                                <tr key={idx}>
                                    <td>{item.iteration}</td>
                                    <td>{item.coherence}</td>
                                    <td>{item.embedding.toFixed(1)}%</td>
                                    <td>{item.llm.toFixed(1)}%</td>
                                    <td>{item.time.toFixed(2)}s</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            );
        }

        function ModelResponses() {
            const [expandedModel, setExpandedModel] = React.useState(null);
            const [expandedQuestion, setExpandedQuestion] = React.useState(null);

            const prepareMetricsData = (answers) => {
                return answers.map(answer => ({
                    iteration: answer.answer_num,
                    coherence: answer.coherence_score,
                    embedding: answer.embedding_dissimilarity_score * 100,
                    llm: answer.llm_dissimilarity_score * 100,
                    time: answer.processing_time
                }));
            };

            return (
                <div className="container mx-auto p-4 max-w-6xl">
                    <h1 className="text-2xl font-bold mb-4">Model Response Analysis</h1>
                    
                    {Object.entries(modelData.models).map(([modelName, modelData]) => (
                        <div key={modelName} className="card">
                            <button 
                                className="w-full text-left font-bold text-lg p-2 hover:bg-gray-50 rounded flex items-center justify-between"
                                onClick={() => setExpandedModel(expandedModel === modelName ? null : modelName)}
                            >
                                <span>{modelName}</span>
                                <span>{expandedModel === modelName ? '▼' : '▶'}</span>
                            </button>
                            
                            {expandedModel === modelName && Object.entries(modelData[0.7]).map(([question, answers]) => (
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
                                                            <div className="metric">
                                                                <span className="font-medium">LLM Dissimilarity:</span> {(answer.llm_dissimilarity_score * 100).toFixed(1)}%
                                                            </div>
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
    if len(sys.argv) != 2:
        print("Usage: python visualize.py <path_to_pickle_file>")
        sys.exit(1)
    
    pkl_path = sys.argv[1]
    create_visualization(pkl_path)
