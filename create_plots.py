# %%
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, NamedTuple, Set
from clusters import question_w_clusters
from question_list import questions
from collections import defaultdict


# Define company colors
COMPANY_COLORS = {
    'openai': '#74AA9C',      # OpenAI green
    'meta-llama': '#044EAB',  # Meta blue
    'anthropic': '#D4C5B9',   # Anthropic beige
    'google': '#669DF7',      # Google blue
    'x-ai': '#000000',        # X black
    'mistralai': '#F54E42'    # Mistral red
}


class ModelMetrics(NamedTuple):
    embedding_total: float
    coherence_total: float
    valid_answers: int

def load_results(file_path: str) -> dict:
    """Load results from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def calculate_metrics(results: dict,
                     min_embedding_threshold: float = 0.15,
                     min_coherence_threshold: float = 15.0) -> Dict[str, Dict[str, ModelMetrics]]:
    """Calculate metrics for each model at each temperature setting."""
    metrics = {}
    
    for model_name, temp_data in results['models'].items():
        metrics[model_name] = {}
        
        for temp, questions in temp_data.items():
            embedding_total = 0
            coherence_total = 0
            valid_answers = 0
            
            for question, answers in questions.items():
                for answer in answers:
                    if (answer['embedding_dissimilarity_score'] >= min_embedding_threshold and 
                        answer['coherence_score'] >= min_coherence_threshold):
                        embedding_total += answer['embedding_dissimilarity_score']
                        coherence_total += answer['coherence_score'] / 100  # Divide coherence by 100
                        valid_answers += 1
            
            metrics[model_name][temp] = ModelMetrics(
                embedding_total=embedding_total,
                coherence_total=coherence_total,
                valid_answers=valid_answers
            )
                
    return metrics

def plot_metric(metrics: Dict[str, Dict[str, ModelMetrics]], 
                metric_name: str,
                metric_getter,
                output_dir: str,
                figure_name: str,
                ylabel: str,
                format_value=lambda x: f"{x:.2f}",
                paper_ready=True,
                color_by_company=True
                ) -> None:
    """
    Create a horizontal bar plot for the specified metric.
    """
    
    models = list(metrics.keys())
    temps = list(metrics[models[0]].keys())
    
    # Calculate max value for each model for ranking
    max_values = {model: max(metric_getter(temp_metrics) 
                           for temp_metrics in temp_data.values())
                 for model, temp_data in metrics.items()}
    
    # Sort models by their max values
    sorted_models = sorted(models, key=lambda x: max_values[x], reverse=True)
    
    # Set up the plot
    plt.figure(figsize=(12, max(6, len(models) * 0.5)))
    
    # Calculate bar positions - note the reverse order
    bar_height = 0.8 / len(temps)
    positions_base = np.arange(len(models))[::-1]  # Reverse the base positions
    
    # Create bars for each temperature
    for i, temp in enumerate(temps):
        positions = positions_base + i * bar_height
        values = [metric_getter(metrics[model][temp]) for model in sorted_models]
        
        if color_by_company:
            # Create bars with company-specific colors
            for pos, val, model in zip(positions, values, sorted_models):
                company = model.split('/')[0]
                bar = plt.barh(pos, val, bar_height, 
                             color=COMPANY_COLORS[company],
                             alpha=0.8)
                # Add value label
                plt.text(val + max(values) * 0.01,  # Slight offset from bar end
                        pos,
                        format_value(val),
                        va='center',
                        fontsize=8)
        else:
            # Create bars with default coloring
            container = plt.barh(positions, values, bar_height, 
                               label=f'Temperature {temp}' if not paper_ready else None,
                               alpha=0.8)
            # Add value labels
            for rect, val in zip(container.patches, values):
                plt.text(val + max(values) * 0.01,  # Slight offset from bar end
                        rect.get_y() + rect.get_height()/2,
                        format_value(val),
                        va='center',
                        fontsize=8)
    
    # Customize the plot
    plt.ylabel('Models')
    plt.xlabel(ylabel)
    if paper_ready:
        plt.title(f'{metric_name} by Model')
    else:
        plt.title(f'{metric_name} by Model and Temperature\n(Filtered for embedding ≥ 0.15 and coherence ≥ 15)')
    
    # Position model names - note they're already in reverse order due to positions_base
    plt.yticks(np.arange(len(models))[::-1] + (bar_height * (len(temps) - 1)) / 2, 
              [m.split('/')[-1] for m in sorted_models])
    
    # Add legend
    if color_by_company:
        # Add company color legend
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, label=company) 
                         for company, color in COMPANY_COLORS.items()]
        plt.legend(handles=legend_elements,
                  bbox_to_anchor=(1.05, 1),
                  loc='upper left')
    elif not paper_ready:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.grid(True, alpha=0.3, axis='x')
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(Path(output_dir) / figure_name, dpi=300, bbox_inches='tight')
    plt.close()

def analyze_benchmark(file_path: str,
                     output_dir: str = 'plots',
                     min_embedding_threshold: float = 0.15,
                     min_coherence_threshold: float = 15.0) -> Dict[str, Dict[str, ModelMetrics]]:
    """Main function to analyze benchmark results and generate plots."""
    results = load_results(file_path)
    metrics = calculate_metrics(
        results,
        min_embedding_threshold=min_embedding_threshold,
        min_coherence_threshold=min_coherence_threshold
    )
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate three different plots with appropriate value formatting
    plot_metric(
        metrics,
        "Total Embedding Dissimilarity",
        lambda x: x.embedding_total,
        output_dir,
        'embedding_scores.png',
        'Total Embedding Dissimilarity Score',
        lambda x: f"{x:.2f}"
    )
    
    plot_metric(
        metrics,
        "Total Coherence Score",
        lambda x: x.coherence_total,
        output_dir,
        'coherence_scores.png',
        'Total Coherence Score',
        lambda x: f"{x:.2f}"
    )
    
    plot_metric(
        metrics,
        "Number of Valid Responses",
        lambda x: x.valid_answers,
        output_dir,
        'valid_responses.png',
        'Number of Valid Responses',
        lambda x: f"{int(x)}"  # Format as integer for count
    )
    
    return metrics


#####

def get_cluster_questions(cluster: str, questions_data: List[dict]) -> Set[str]:
    """Get all questions belonging to a specific cluster."""
    return {q['question'] for q in questions_data if cluster in q['clusters']}

def calculate_cluster_metrics(results: dict,
                            cluster_questions: Set[str],
                            min_embedding_threshold: float = 0.15,
                            min_coherence_threshold: float = 15.0) -> Dict[str, Dict[str, ModelMetrics]]:
    """Calculate metrics for each model at each temperature setting for specific cluster questions."""
    metrics = {}
    
    for model_name, temp_data in results['models'].items():
        metrics[model_name] = {}
        
        for temp, questions in temp_data.items():
            embedding_total = 0
            coherence_total = 0
            valid_answers = 0
            
            # Only process questions that belong to the cluster
            for question in cluster_questions:
                if question in questions:
                    for answer in questions[question]:
                        if (answer['embedding_dissimilarity_score'] >= min_embedding_threshold and 
                            answer['coherence_score'] >= min_coherence_threshold):
                            embedding_total += answer['embedding_dissimilarity_score']
                            coherence_total += answer['coherence_score'] / 100
                            valid_answers += 1
            
            metrics[model_name][temp] = ModelMetrics(
                embedding_total=embedding_total,
                coherence_total=coherence_total,
                valid_answers=valid_answers
            )
                
    return metrics

def plot_cluster_metrics(results: dict,
                        questions_data: List[dict],
                        clusters: List[str],
                        output_dir: str = 'plots',
                        min_embedding_threshold: float = 0.15,
                        min_coherence_threshold: float = 15.0) -> None:
    """Generate plots for each cluster."""
    
    # Create cluster-specific directory
    cluster_dir = Path(output_dir) / 'clusters'
    cluster_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each cluster
    for cluster in clusters:
        print(f"Processing cluster: {cluster}")
        
        # Get questions for this cluster
        cluster_questions = get_cluster_questions(cluster, questions_data)
        
        # Skip if no questions in cluster
        if not cluster_questions:
            print(f"No questions found for cluster: {cluster}")
            continue
        
        # Calculate metrics for this cluster
        metrics = calculate_cluster_metrics(
            results,
            cluster_questions,
            min_embedding_threshold,
            min_coherence_threshold
        )
        
        # Create cluster-specific subdirectory
        cluster_subdir = cluster_dir / cluster.lower().replace(' ', '_')
        cluster_subdir.mkdir(parents=True, exist_ok=True)
        
        # Generate the three plots for this cluster
        plot_metric(
            metrics,
            f"Total Embedding Dissimilarity - {cluster}",
            lambda x: x.embedding_total,
            cluster_subdir,
            'embedding_scores.png',
            'Total Embedding Dissimilarity Score',
            lambda x: f"{x:.2f}"
        )
        
        plot_metric(
            metrics,
            f"Total Coherence Score - {cluster}",
            lambda x: x.coherence_total,
            cluster_subdir,
            'coherence_scores.png',
            'Total Coherence Score',
            lambda x: f"{x:.2f}"
        )
        
        plot_metric(
            metrics,
            f"Number of Valid Responses - {cluster}",
            lambda x: x.valid_answers,
            cluster_subdir,
            'valid_responses.png',
            'Number of Valid Responses',
            lambda x: f"{int(x)}"
        )

def analyze_clusters(results_file: str,
                    questions_data: List[dict],  # Now takes Python list directly
                    output_dir: str = 'plots',
                    min_embedding_threshold: float = 0.15,
                    min_coherence_threshold: float = 15.0) -> None:
    """Main function to analyze benchmark results by clusters."""
    
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Extract unique clusters
    clusters = set()
    for question in questions_data:
        clusters.update(question['clusters'])
    clusters = sorted(list(clusters))
    
    # Generate plots for each cluster
    plot_cluster_metrics(
        results,
        questions_data,
        clusters,
        output_dir,
        min_embedding_threshold,
        min_coherence_threshold
    )
    
    print("\nCluster analysis complete. Plots have been generated in the 'plots/clusters' directory.")



#####

def get_company_from_model(model_name: str) -> str:
    """Extract company name from model name."""
    return model_name.split('/')[0]

def get_best_models_per_cluster(results: dict,
                              questions_data: List[dict],
                              min_embedding_threshold: float = 0.15,
                              min_coherence_threshold: float = 15.0) -> Dict[str, Dict[str, Tuple[str, float]]]:
    """
    For each cluster, find the best performing model for each metric.
    Scores are averaged by the number of questions in each cluster.
    Returns dict[cluster][metric] = (model_name, average_score)
    """
    # Get unique clusters and count questions per cluster
    clusters = set()
    cluster_question_counts = defaultdict(int)
    for question in questions_data:
        for cluster in question['clusters']:
            clusters.add(cluster)
            cluster_question_counts[cluster] += 1
    
    # Initialize results storage
    best_performers = {cluster: {} for cluster in clusters}
    
    # Process each cluster
    for cluster in clusters:
        # Get questions for this cluster
        cluster_questions = {q['question'] for q in questions_data if cluster in q['clusters']}
        num_questions = cluster_question_counts[cluster]
        
        # Initialize metric tracking for this cluster
        metrics = defaultdict(lambda: defaultdict(float))
        
        # Calculate metrics for each model
        for model_name, temp_data in results['models'].items():
            for temp, questions in temp_data.items():
                embedding_total = 0
                coherence_total = 0
                valid_answers = 0
                
                for question in cluster_questions:
                    if question in questions:
                        for answer in questions[question]:
                            if (answer['embedding_dissimilarity_score'] >= min_embedding_threshold and 
                                answer['coherence_score'] >= min_coherence_threshold):
                                embedding_total += answer['embedding_dissimilarity_score']
                                coherence_total += answer['coherence_score'] / 100
                                valid_answers += 1
                
                # Average scores by number of questions in cluster
                avg_embedding = embedding_total / num_questions if num_questions > 0 else 0
                avg_coherence = coherence_total / num_questions if num_questions > 0 else 0
                avg_answers = valid_answers / num_questions if num_questions > 0 else 0
                
                # Update best scores if this temperature setting is better
                metrics['embedding'][model_name] = max(metrics['embedding'][model_name], avg_embedding)
                metrics['coherence'][model_name] = max(metrics['coherence'][model_name], avg_coherence)
                metrics['answers'][model_name] = max(metrics['answers'][model_name], avg_answers)
        
        # Find best model for each metric
        for metric_name, model_scores in metrics.items():
            if model_scores:
                best_model = max(model_scores.items(), key=lambda x: x[1])
                best_performers[cluster][metric_name] = best_model
    
    return best_performers

def plot_best_performers(best_performers: Dict[str, Dict[str, Tuple[str, float]]],
                        metric: str,
                        output_dir: str = 'plots',
                        title_suffix: str = '') -> None:
    """Create a horizontal bar plot for best performers in a specific metric."""
    
    # Prepare data
    clusters = list(best_performers.keys())
    models = [best_performers[cluster][metric][0] if metric in best_performers[cluster] else None for cluster in clusters]
    scores = [best_performers[cluster][metric][1] if metric in best_performers[cluster] else 0 for cluster in clusters]
    
    # Sort by score
    sorted_indices = np.argsort(scores)
    clusters = [clusters[i] for i in sorted_indices]
    models = [models[i] for i in sorted_indices]
    scores = [scores[i] for i in sorted_indices]
    
    # Create figure
    plt.figure(figsize=(12, max(8, len(clusters) * 0.4)))
    
    # Create bars
    bars = plt.barh(range(len(clusters)), scores, height=0.7)
    
    # Color bars by company and add model names
    for idx, (bar, model) in enumerate(zip(bars, models)):
        if model:
            company = get_company_from_model(model)
            bar.set_color(COMPANY_COLORS[company])
            
            # Add model name inside bar
            model_name = model.split('/')[-1]  # Get just the model name part
            x_pos = bar.get_width() * 0.02  # Small offset from start of bar
            plt.text(x_pos, idx, f"{model_name} ({scores[idx]:.2f})", 
                    va='center', fontsize=8, color='black')
    
    # Customize plot
    plt.ylabel('Clusters')
    xlabel = {
        'embedding': 'Average Embedding Dissimilarity Score per Question',
        'coherence': 'Average Coherence Score per Question',
        'answers': 'Average Valid Responses per Question'
    }[metric]
    plt.xlabel(xlabel)
    
    plt.title(f'Best Performing Models by {xlabel}{title_suffix}')
    plt.yticks(range(len(clusters)), [c.replace(' and ', '\n& ') for c in clusters], fontsize=8)
    
    # Add legend for companies
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, label=company) 
                      for company, color in COMPANY_COLORS.items()]
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Save plot
    plt.tight_layout()
    plt.savefig(Path(output_dir) / f'best_{metric}_per_cluster_averaged.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def analyze_best_performers(results_file: str,
                          questions_data: List[dict],
                          output_dir: str = 'plots') -> None:
    """Generate best performer analysis and plots."""
    
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Get best performers
    best_performers = get_best_models_per_cluster(
        results,
        questions_data,
        min_embedding_threshold=0.15,
        min_coherence_threshold=15.0
    )
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate plots for each metric
    for metric in ['embedding', 'coherence', 'answers']:
        plot_best_performers(best_performers, metric, output_dir)
        
    return best_performers



#####


def normalize_text(text: str) -> str:
    """Normalize text for comparison by removing extra spaces and hyphens."""
    return text.lower().replace('-', ' ').replace('  ', ' ').strip()

def get_best_models_per_question(results: dict,
                               questions_data: List[dict],
                               min_embedding_threshold: float = 0.15,
                               min_coherence_threshold: float = 15.0) -> Dict[str, Dict[int, Tuple[str, float]]]:
    """
    For each metric, find the best performing model for each question.
    Scores are summed across all valid answers for each model-question pair.
    Returns dict[metric][question_num] = (model_name, score)
    """
    # Initialize storage
    best_scores = {
        'embedding': {},
        'coherence': {},
        'answers': {}
    }
    
    # Create normalized question map
    question_map = {normalize_text(q['question']): q['number'] for q in questions_data}
    
    # First, calculate total scores for each model-question pair
    model_scores = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    
    for model_name, temp_data in results['models'].items():
        for temp, questions in temp_data.items():
            for question, answers in questions.items():
                normalized_q = normalize_text(question)
                if normalized_q not in question_map:
                    print(f"Warning: Could not find matching question for: {question}")
                    continue
                    
                q_num = question_map[normalized_q]
                
                # Initialize scores for this model-temp-question combination
                embedding_total = 0
                coherence_total = 0
                valid_count = 0
                
                # Sum up scores across all valid answers
                for answer in answers:
                    if (answer['embedding_dissimilarity_score'] >= min_embedding_threshold and 
                        answer['coherence_score'] >= min_coherence_threshold):
                        embedding_total += answer['embedding_dissimilarity_score']
                        coherence_total += answer['coherence_score'] / 100
                        valid_count += 1
                
                # Update scores if this temperature setting gives better results
                current_embedding = model_scores['embedding'][model_name][q_num]
                current_coherence = model_scores['coherence'][model_name][q_num]
                current_answers = model_scores['answers'][model_name][q_num]
                
                model_scores['embedding'][model_name][q_num] = max(current_embedding, embedding_total)
                model_scores['coherence'][model_name][q_num] = max(current_coherence, coherence_total)
                model_scores['answers'][model_name][q_num] = max(current_answers, valid_count)
    
    # Now find the best model for each question
    for metric in ['embedding', 'coherence', 'answers']:
        for q_num in question_map.values():
            best_score = -float('inf')
            best_model = None
            
            for model_name in results['models'].keys():
                score = model_scores[metric][model_name][q_num]
                if score > best_score:
                    best_score = score
                    best_model = model_name
            
            if best_model is not None:
                best_scores[metric][q_num] = (best_model, best_score)
    
    return best_scores


def wrap_text(text: str, width: int = 15) -> str:
    """Wrap text to specified width, preserving words."""
    words = text.replace(' and ', '\n& ').split()
    lines = []
    current_line = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) + 1 <= width:
            current_line.append(word)
            current_length += len(word) + 1
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
            current_length = len(word)
    
    if current_line:
        lines.append(' '.join(current_line))
    
    return '\n'.join(lines)

def plot_question_performance(best_scores: Dict[str, Dict[int, Tuple[str, float]]],
                            questions_data: List[dict],
                            metric: str,
                            output_dir: str = 'plots',
                            show_question_numbers: bool = True) -> None:
    """Create a hierarchical horizontal bar plot for question-level performance."""
    
    # Group questions by cluster
    cluster_questions = defaultdict(list)
    for q in questions_data:
        for cluster in q['clusters']:
            cluster_questions[cluster].append(q['number'])
    
    # Sort and organize data
    organized_data = []
    y_labels = []
    scores = []
    models = []
    
    for cluster in sorted(cluster_questions.keys()):
        questions = cluster_questions[cluster]
        # Sort questions by their best score within cluster
        sorted_questions = sorted(
            questions,
            key=lambda q: best_scores[metric][q][1] if q in best_scores[metric] else -float('inf'),
            reverse=True
        )
        
        # Add cluster label as placeholder
        organized_data.append((cluster, None, None))
        
        # Add questions
        for q in sorted_questions:
            if q in best_scores[metric]:
                organized_data.append((None, q, best_scores[metric][q]))
    
    # Create figure with GridSpec for better control
    fig = plt.figure(figsize=(12, max(8, len(organized_data) * 0.3)))
    gs = fig.add_gridspec(1, 2, width_ratios=[2, 15], wspace=0.01)  # Increased width ratio for labels
    
    # Create two axes: one for labels, one for the main plot
    ax_labels = fig.add_subplot(gs[0])
    ax_main = fig.add_subplot(gs[1])
    
    # Process data and create bars
    current_y = 0
    y_positions = []
    cluster_positions = []
    cluster_heights = []
    
    for item in organized_data:
        cluster, q_num, score_data = item
        if cluster:  # This is a cluster label
            cluster_positions.append(current_y)
            current_y += 0.5  # Add some space before first question
        else:  # This is a question
            y_positions.append(current_y)
            model, score = score_data
            scores.append(score)
            models.append(model)
            current_y += 1
    
    # Calculate cluster heights
    for i in range(len(cluster_positions)):
        if i < len(cluster_positions) - 1:
            cluster_heights.append(cluster_positions[i+1] - cluster_positions[i] - 0.5)
        else:
            cluster_heights.append(current_y - cluster_positions[i] - 0.5)
    
    # Plot bars
    bars = ax_main.barh(y_positions, scores, height=0.7)
    
    # Color bars and add labels
    for idx, (bar, model) in enumerate(zip(bars, models)):
        company = model.split('/')[0]
        bar.set_color(COMPANY_COLORS[company])
        
        # Add model name inside bar
        model_name = model.split('/')[-1]
        x_pos = bar.get_width() * 0.02
        ax_main.text(x_pos, y_positions[idx], f"{model_name} ({scores[idx]:.2f})", 
                    va='center', fontsize=6, color='black')
    
    # Add cluster labels and backgrounds
    for cluster_y, height, (cluster, _, _) in zip(cluster_positions, cluster_heights, 
                                                [x for x in organized_data if x[0]]):
        # Add wrapped cluster label
        wrapped_cluster = wrap_text(cluster)
        ax_labels.text(0.8, cluster_y + height/2, wrapped_cluster,
                      ha='right', va='center',
                      fontsize=8, fontweight='bold',
                      linespacing=0.9)
        
        # Add alternating background
        ax_main.axhspan(cluster_y, cluster_y + height,
                       color='gray' if cluster_positions.index(cluster_y) % 2 == 0 else 'white',
                       alpha=0.1)
    
    # Add question numbers on the left
    question_positions = []
    question_labels = []
    for item in organized_data:
        if not item[0]:  # This is a question
            question_positions.append(y_positions[len(question_labels)])
            # question_labels.append(f"Q{item[1]}")
            question_labels.append(" ")
    
    ax_labels.set_ylim(ax_main.get_ylim())
    ax_labels.set_xlim(-1, 1)
    
    # Set y-tick labels for questions
    ax_labels.set_yticks(question_positions)
    ax_labels.set_yticklabels(question_labels, fontsize=8)
    
    # Move question numbers to the far left
    ax_labels.set_yticklabels(question_labels, fontsize=8)
    for tick in ax_labels.yaxis.get_major_ticks():
        tick.set_pad(-20)  # Adjust the padding to move numbers closer to the left edge
        tick.tick1line.set_visible(False)  # Hide the tick marks
        tick.tick2line.set_visible(False)  # Hide the tick marks on the other side
    
    # Customize axes
    ax_labels.set_xticks([])
    ax_labels.spines['right'].set_visible(False)
    ax_labels.spines['top'].set_visible(False)
    ax_labels.spines['bottom'].set_visible(False)
    ax_labels.spines['left'].set_visible(False)
    
    ax_main.set_yticks([])
    
    # Set title and labels
    xlabel = {
        'embedding': 'Total Embedding Dissimilarity Score',
        'coherence': 'Total Coherence Score',
        'answers': 'Number of Valid Responses'
    }[metric]
    ax_main.set_xlabel(xlabel)
    
    fig.suptitle(f'Best Performance by Question\nGrouped by Cluster, Ordered by {xlabel}',
                 y=1.02)
    
    # Add legend for companies
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, label=company) 
                      for company, color in COMPANY_COLORS.items()]
    ax_main.legend(handles=legend_elements, 
                  loc='upper center',
                  bbox_to_anchor=(0.5, 1.15),
                  ncol=len(COMPANY_COLORS))
    
    # Save plot
    plt.savefig(Path(output_dir) / f'question_level_{metric}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()


def analyze_question_performance(results_file: str,
                               questions_data: List[dict],
                               output_dir: str = 'plots') -> None:
    """Generate question-level analysis and plots."""
    
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Get best performers for each question
    best_scores = get_best_models_per_question(
        results,
        questions_data,
        min_embedding_threshold=0.15,
        min_coherence_threshold=15.0
    )
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate plots for each metric
    for metric in ['embedding', 'coherence', 'answers']:
        plot_question_performance(best_scores, questions_data, metric, output_dir)
        
    return best_scores

