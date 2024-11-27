# %%
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, NamedTuple, Set
from clusters import question_w_clusters, wordcel_questions, shape_rotator_questions
from question_list import questions
from collections import defaultdict
from model_list import lmsys_scores, release_dates, model_scales

from scipy import stats
import adjustText

from datetime import datetime
import matplotlib.dates as mdates

import csv
import pandas as pd


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



#####


def get_max_scores(results: dict,
                  min_embedding_threshold: float = 0.15,
                  min_coherence_threshold: float = 15.0) -> Dict[str, Dict[str, float]]:
    """
    Calculate maximum scores for each model across temperatures.
    """
    max_scores = defaultdict(lambda: {'embedding': 0, 'coherence': 0, 'answers': 0})
    
    for model_name, temp_data in results['models'].items():
        for temp, questions in temp_data.items():
            embedding_total = 0
            coherence_total = 0
            valid_answers = 0
            
            for answers in questions.values():
                for answer in answers:
                    if (answer['embedding_dissimilarity_score'] >= min_embedding_threshold and 
                        answer['coherence_score'] >= min_coherence_threshold):
                        embedding_total += answer['embedding_dissimilarity_score']
                        coherence_total += answer['coherence_score'] / 100
                        valid_answers += 1
            
            # Update maximum scores
            max_scores[model_name]['embedding'] = max(max_scores[model_name]['embedding'], 
                                                    embedding_total)
            max_scores[model_name]['coherence'] = max(max_scores[model_name]['coherence'], 
                                                    coherence_total)
            max_scores[model_name]['answers'] = max(max_scores[model_name]['answers'], 
                                                  valid_answers)
    
    return max_scores

def create_correlation_plots(results: dict,
                           lmsys_data: List[dict],
                           output_dir: str = 'plots',
                           show_correlation: bool = True) -> None:
    """
    Create scatter plots comparing LMSYS scores with benchmark metrics.
    """
    
    # Get maximum scores for each model
    max_scores = get_max_scores(results)
    
    # Convert LMSYS data to dict for easier lookup
    lmsys_dict = {item['model']: item['lmsys_score'] for item in lmsys_data}
    
    # Metrics to plot
    metrics_to_plot = {
        'embedding': 'Embedding Dissimilarity Score',
        'coherence': 'Coherence Score',
        'answers': 'Valid Responses'
    }
    
    for metric_key, metric_name in metrics_to_plot.items():
        # Collect data points
        plot_data = []
        for model, scores in max_scores.items():
            if model in lmsys_dict:
                plot_data.append({
                    'model': model,
                    'benchmark_score': scores[metric_key],
                    'lmsys_score': lmsys_dict[model],
                    'company': model.split('/')[0]
                })
        
        if not plot_data:
            continue
            
        # Create scatter plot
        plt.figure(figsize=(12, 8))
        
        # Plot points and prepare correlation data
        xs = [d['lmsys_score'] for d in plot_data]
        ys = [d['benchmark_score'] for d in plot_data]
        
        # Calculate correlation if requested
        if show_correlation and len(xs) > 1:
            correlation, p_value = stats.pearsonr(xs, ys)
            spearman_corr, spearman_p = stats.spearmanr(xs, ys)
        
        # Create scatter plot with company colors
        for data in plot_data:
            plt.scatter(data['lmsys_score'], data['benchmark_score'],
                       color=COMPANY_COLORS[data['company']],
                       s=100, alpha=0.7)
        
        # Add labels with smart placement
        texts = []
        for data in plot_data:
            model_name = data['model'].split('/')[-1]
            texts.append(plt.text(data['lmsys_score'], data['benchmark_score'], 
                                model_name, fontsize=8))
        
        # Adjust text positions to avoid overlap
        adjustText.adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5))
        
        # Add correlation information if requested
        if show_correlation and len(xs) > 1:
            correlation_text = f"Pearson Correlation: {correlation:.3f}\n"
            correlation_text += f"Spearman Correlation: {spearman_corr:.3f}"
            plt.text(0.05, 0.95, correlation_text,
                    transform=plt.gca().transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Customize plot
        plt.xlabel('LMSYS Score')
        plt.ylabel(f'AidanBench {metric_name}')
        plt.title(f'LMSYS Score vs AidanBench {metric_name}')
        
        # Add legend for companies
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=color, label=company, markersize=10)
                         for company, color in COMPANY_COLORS.items()]
        plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
        
        # Add grid
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.tight_layout()
        plt.savefig(Path(output_dir) / f'lmsys_correlation_{metric_key}.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()

def analyze_lmsys_correlation(results_file: str,
                            lmsys_data: List[dict],
                            output_dir: str = 'plots',
                            show_correlation: bool = True) -> None:
    """Main function to analyze LMSYS correlations and create plots."""
    
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate correlation plots
    create_correlation_plots(results, lmsys_data, output_dir, show_correlation)



#####


def create_timeline_plots(results: dict,
                        release_dates: List[dict],
                        output_dir: str = 'plots') -> None:
    """
    Create scatter plots showing score evolution over time.
    """
    
    # Get maximum scores for each model
    max_scores = get_max_scores(results)
    
    # Convert release dates to dict and datetime objects
    release_dict = {item['model']: datetime.strptime(item['release_date'], '%Y-%m-%d')
                   for item in release_dates}
    
    # Metrics to plot
    metrics_to_plot = {
        'embedding': 'Embedding Dissimilarity Score',
        'coherence': 'Coherence Score',
        'answers': 'Valid Responses'
    }
    
    for metric_key, metric_name in metrics_to_plot.items():
        # Collect data points
        plot_data = []
        for model, scores in max_scores.items():
            if model in release_dict:
                plot_data.append({
                    'model': model,
                    'score': scores[metric_key],
                    'date': release_dict[model],
                    'company': model.split('/')[0]
                })
        
        if not plot_data:
            continue
            
        # Create scatter plot
        plt.figure(figsize=(15, 8))
        
        # Plot points
        for data in plot_data:
            plt.scatter(data['date'], data['score'],
                       color=COMPANY_COLORS[data['company']],
                       s=100, alpha=0.7)
        
        # Add labels with smart placement
        texts = []
        for data in plot_data:
            model_name = data['model'].split('/')[-1]
            texts.append(plt.text(data['date'], data['score'], 
                                model_name, fontsize=8))
        
        # Adjust text positions to avoid overlap
        adjustText.adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5))
        
        # Customize plot
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        
        # Rotate and align the tick labels so they look better
        plt.gcf().autofmt_xdate()
        
        # # Add trend line
        # dates = mdates.date2num([d['date'] for d in plot_data])
        # scores = [d['score'] for d in plot_data]
        # z = np.polyfit(dates, scores, 1)
        # p = np.poly1d(z)
        
        # Add trend line
        # plt.plot(
        #     [min(dates), max(dates)],
        #     [p(min(dates)), p(max(dates))],
        #     "k--", alpha=0.5, label=f'Trend line'
        # )
        
        plt.xlabel('Release Date')
        plt.ylabel(f'Benchmark {metric_name}')
        plt.title(f'Evolution of {metric_name} Over Time')
        
        # Add legend for companies
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=color, label=company, markersize=10)
                         for company, color in COMPANY_COLORS.items()]
        legend_elements.append(plt.Line2D([0], [0], linestyle='--', color='k', alpha=0.5,
                                        label='Trend line'))
        plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
        
        # Add grid
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.tight_layout()
        plt.savefig(Path(output_dir) / f'timeline_{metric_key}.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()

def analyze_timeline(results_file: str,
                    release_dates: List[dict],
                    output_dir: str = 'plots') -> None:
    """Main function to analyze score evolution over time."""
    
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate timeline plots
    create_timeline_plots(results, release_dates, output_dir)



#####

def calculate_scores_for_questions(results: dict,
                                 questions: List[str],
                                 min_embedding_threshold: float = 0.15,
                                 min_coherence_threshold: float = 15.0) -> Dict[str, Dict[str, float]]:
    """
    Calculate scores for specific questions.
    Returns dict[model_name][metric] = score
    """
    scores = defaultdict(lambda: {'embedding': 0, 'coherence': 0, 'answers': 0})
    
    for model_name, temp_data in results['models'].items():
        model_max_scores = defaultdict(float)
        
        for temp, question_data in temp_data.items():
            embedding_total = 0
            coherence_total = 0
            valid_answers = 0
            
            for question in questions:
                if question in question_data:
                    for answer in question_data[question]:
                        if (answer['embedding_dissimilarity_score'] >= min_embedding_threshold and 
                            answer['coherence_score'] >= min_coherence_threshold):
                            embedding_total += answer['embedding_dissimilarity_score']
                            coherence_total += answer['coherence_score'] / 100
                            valid_answers += 1
            
            # Divide by number of questions to get average
            if questions:  # Avoid division by zero
                embedding_total /= len(questions)
                coherence_total /= len(questions)
                valid_answers /= len(questions)
            
            # Update maximum scores for this temperature
            model_max_scores['embedding'] = max(model_max_scores['embedding'], embedding_total)
            model_max_scores['coherence'] = max(model_max_scores['coherence'], coherence_total)
            model_max_scores['answers'] = max(model_max_scores['answers'], valid_answers)
        
        # Store best scores across temperatures
        scores[model_name] = dict(model_max_scores)
    
    return scores

def create_question_set_comparison(results: dict,
                                 questions_set1: List[str],
                                 questions_set2: List[str],
                                 output_dir: str = 'plots',
                                 normalize_scales: bool = False) -> None:
    """
    Create scatter plots comparing model performance between two sets of questions.
    
    Args:
        results: Results dictionary
        questions_set1: First set of questions
        questions_set2: Second set of questions
        output_dir: Directory to save plots
        normalize_scales: If True, rescale axes to make best-fit line appear as y=x
    """    
    # Calculate scores for both sets
    scores_set1 = calculate_scores_for_questions(results, questions_set1)
    scores_set2 = calculate_scores_for_questions(results, questions_set2)
    
    # Metrics to plot
    metrics_to_plot = {
        'embedding': 'Embedding Dissimilarity Score',
        'coherence': 'Coherence Score',
        'answers': 'Valid Responses'
    }
    
    for metric_key, metric_name in metrics_to_plot.items():
        # Collect data points
        plot_data = []
        for model in results['models'].keys():
            if model in scores_set1 and model in scores_set2:
                plot_data.append({
                    'model': model,
                    'score1': scores_set1[model][metric_key],
                    'score2': scores_set2[model][metric_key],
                    'company': model.split('/')[0]
                })
        
        if not plot_data:
            continue
            
        # Create scatter plot
        plt.figure(figsize=(12, 8))
        
        # Get scores for calculations
        scores1 = np.array([d['score1'] for d in plot_data])
        scores2 = np.array([d['score2'] for d in plot_data])
        
        if normalize_scales:
            # Calculate scaling factors to make best-fit line appear as y=x
            # Use linear regression to find the relationship
            z = np.polyfit(scores1, scores2, 1)
            slope, intercept = z
            
            # Scale scores2 to match scores1's scale
            scores2_scaled = (scores2 - intercept) / slope
            
            # Update plot data with scaled scores
            for data, scaled_score2 in zip(plot_data, scores2_scaled):
                data['score2_scaled'] = scaled_score2
            
            # Use scaled scores for plotting
            max_score = max(max(scores1), max(scores2_scaled))
            plot_scores2 = scores2_scaled
        else:
            max_score = max(max(scores1), max(scores2))
            plot_scores2 = scores2
        
        # Plot diagonal line
        plt.plot([0, max_score * 1.1], [0, max_score * 1.1], 
                 'k--', alpha=0.3, label='y=x')
        
        # Plot points
        for idx, data in enumerate(plot_data):
            plt.scatter(data['score1'], 
                       data['score2_scaled'] if normalize_scales else data['score2'],
                       color=COMPANY_COLORS[data['company']],
                       s=100, alpha=0.7)
        
        # Add labels with smart placement
        texts = []
        for data in plot_data:
            model_name = data['model'].split('/')[-1]
            texts.append(plt.text(data['score1'], 
                                data['score2_scaled'] if normalize_scales else data['score2'], 
                                model_name, fontsize=8))
        
        # Adjust text positions to avoid overlap
        adjustText.adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5))
        
        # Calculate correlation
        correlation = np.corrcoef(scores1, scores2)[0, 1]
        
        # Add correlation text
        plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
                transform=plt.gca().transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Set equal aspect ratio and limits
        plt.axis('equal')
        plt.xlim(0, max_score * 1.1)
        plt.ylim(0, max_score * 1.1)
        
        # Labels and title
        plt.xlabel(f'Average {metric_name} on wordcel questions')
        plt.ylabel(f'Average {metric_name} on shape rotator questions')
        plt.title(f'Model Performance Comparison:\n{metric_name}')
        
        # if normalize_scales:
        #     plt.text(0.05, 0.89, f'Scale factor: {slope:.3f}x + {intercept:.3f}',
        #             transform=plt.gca().transAxes,
        #             verticalalignment='top',
        #             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add legend for companies
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=color, label=company, markersize=10)
                         for company, color in COMPANY_COLORS.items()]
        plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
        
        # Add grid
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.tight_layout()
        suffix = '_normalized' if normalize_scales else ''
        plt.savefig(Path(output_dir) / f'question_set_comparison_{metric_key}{suffix}.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()

def analyze_question_sets(results_file: str,
                         questions_set1: List[str],
                         questions_set2: List[str],
                         output_dir: str = 'plots',
                         normalize_scales: bool = False) -> None:
    """Main function to analyze performance between question sets."""
    
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate comparison plots
    create_question_set_comparison(results, questions_set1, questions_set2, output_dir, normalize_scales)


#####


def get_questions_for_clusters(questions_data: List[dict], clusters: List[str]) -> List[str]:
    """Get all questions that belong to any of the specified clusters."""
    questions = set()
    for q in questions_data:
        if any(cluster in q['clusters'] for cluster in clusters):
            questions.add(q['question'])
    return list(questions)

def calculate_scores_for_clusters(results: dict,
                                questions_data: List[dict],
                                clusters: List[str],
                                min_embedding_threshold: float = 0.15,
                                min_coherence_threshold: float = 15.0) -> Dict[str, Dict[str, float]]:
    """
    Calculate average scores for questions in specified clusters.
    Returns dict[model_name][metric] = score
    """
    # Get questions belonging to these clusters
    cluster_questions = get_questions_for_clusters(questions_data, clusters)
    
    scores = defaultdict(lambda: {'embedding': 0, 'coherence': 0, 'answers': 0})
    
    for model_name, temp_data in results['models'].items():
        model_max_scores = defaultdict(float)
        
        for temp, question_data in temp_data.items():
            embedding_total = 0
            coherence_total = 0
            valid_answers = 0
            
            for question in cluster_questions:
                if question in question_data:
                    for answer in question_data[question]:
                        if (answer['embedding_dissimilarity_score'] >= min_embedding_threshold and 
                            answer['coherence_score'] >= min_coherence_threshold):
                            embedding_total += answer['embedding_dissimilarity_score']
                            coherence_total += answer['coherence_score'] / 100
                            valid_answers += 1
            
            # Divide by number of questions to get average
            if cluster_questions:  # Avoid division by zero
                embedding_total /= len(cluster_questions)
                coherence_total /= len(cluster_questions)
                valid_answers /= len(cluster_questions)
            
            # Update maximum scores for this temperature
            model_max_scores['embedding'] = max(model_max_scores['embedding'], embedding_total)
            model_max_scores['coherence'] = max(model_max_scores['coherence'], coherence_total)
            model_max_scores['answers'] = max(model_max_scores['answers'], valid_answers)
        
        # Store best scores across temperatures
        scores[model_name] = dict(model_max_scores)
    
    return scores

def create_cluster_comparison(results: dict,
                            questions_data: List[dict],
                            clusters_set1: List[str],
                            clusters_set2: List[str],
                            output_dir: str = 'plots') -> None:
    """
    Create scatter plots comparing model performance between two sets of clusters.
    """
    
    # Calculate scores for both sets
    scores_set1 = calculate_scores_for_clusters(results, questions_data, clusters_set1)
    scores_set2 = calculate_scores_for_clusters(results, questions_data, clusters_set2)
    
    # Metrics to plot
    metrics_to_plot = {
        'embedding': 'Embedding Dissimilarity Score',
        'coherence': 'Coherence Score',
        'answers': 'Valid Responses'
    }
    
    for metric_key, metric_name in metrics_to_plot.items():
        # Collect data points
        plot_data = []
        for model in results['models'].keys():
            if model in scores_set1 and model in scores_set2:
                plot_data.append({
                    'model': model,
                    'score1': scores_set1[model][metric_key],
                    'score2': scores_set2[model][metric_key],
                    'company': model.split('/')[0]
                })
        
        if not plot_data:
            continue
            
        # Create scatter plot
        plt.figure(figsize=(12, 8))
        
        # Get max values for axis limits
        max_score = max(
            max(d['score1'] for d in plot_data),
            max(d['score2'] for d in plot_data)
        )
        
        # Plot diagonal line
        plt.plot([0, max_score * 1.1], [0, max_score * 1.1], 
                 'k--', alpha=0.3, label='y=x')
        
        # Plot points
        for data in plot_data:
            plt.scatter(data['score1'], data['score2'],
                       color=COMPANY_COLORS[data['company']],
                       s=100, alpha=0.7)
        
        # Add labels with smart placement
        texts = []
        for data in plot_data:
            model_name = data['model'].split('/')[-1]
            texts.append(plt.text(data['score1'], data['score2'], 
                                model_name, fontsize=8))
        
        # Adjust text positions to avoid overlap
        adjustText.adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5))
        
        # Calculate correlation
        scores1 = [d['score1'] for d in plot_data]
        scores2 = [d['score2'] for d in plot_data]
        correlation = np.corrcoef(scores1, scores2)[0, 1]
        
        # Add correlation text
        plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
                transform=plt.gca().transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Set equal aspect ratio and limits
        plt.axis('equal')
        plt.xlim(0, max_score * 1.1)
        plt.ylim(0, max_score * 1.1)
        
        # Create cluster set descriptions
        set1_desc = ', '.join(clusters_set1)
        set2_desc = ', '.join(clusters_set2)
        
        # Labels and title
        plt.xlabel(f'Average {metric_name} on\n{set1_desc}')
        plt.ylabel(f'Average {metric_name} on\n{set2_desc}')
        plt.title(f'Model Performance Comparison:\n{metric_name}')
        
        # Add legend for companies
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=color, label=company, markersize=10)
                         for company, color in COMPANY_COLORS.items()]
        plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
        
        # Add grid
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.tight_layout()
        plt.savefig(Path(output_dir) / f'cluster_comparison_{metric_key}.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()

def analyze_cluster_sets(results_file: str,
                    questions_data: List[dict],
                    clusters_set1: List[str],
                    clusters_set2: List[str],
                    output_dir: str = 'plots') -> None:
    """Main function to analyze performance between cluster sets. Input lists of cluster names."""
    
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate comparison plots
    create_cluster_comparison(results, questions_data, clusters_set1, clusters_set2, output_dir)


#####


def create_parameter_plots(results: dict,
                         model_scales: List[dict],
                         output_dir: str = 'plots') -> None:
    """
    Create scatter plots comparing model performance versus parameter count.
    """
    
    # Get maximum scores for each model
    max_scores = get_max_scores(results)
    
    # Convert model scales to dict for easier lookup
    scale_dict = {item['model']: item['parameters'] for item in model_scales}
    
    # Metrics to plot
    metrics_to_plot = {
        'embedding': 'Embedding Dissimilarity Score',
        'coherence': 'Coherence Score',
        'answers': 'Valid Responses'
    }
    
    for metric_key, metric_name in metrics_to_plot.items():
        # Collect data points
        plot_data = []
        for model, scores in max_scores.items():
            if model in scale_dict:
                plot_data.append({
                    'model': model,
                    'score': scores[metric_key],
                    'parameters': scale_dict[model],
                    'company': model.split('/')[0]
                })
        
        if not plot_data:
            continue
            
        # Create scatter plot with extra space for labels
        plt.figure(figsize=(12, 8))
        
        # Set logarithmic scale for x-axis early
        plt.xscale('log')
        
        # Plot points
        for data in plot_data:
            plt.scatter(data['parameters'], data['score'],
                       color=COMPANY_COLORS[data['company']],
                       s=100, alpha=0.7)
        
        # Add trend line before labels
        params = [d['parameters'] for d in plot_data]
        scores = [d['score'] for d in plot_data]
        log_params = np.log10(params)
        
        # Calculate correlation with log parameters
        correlation = np.corrcoef(log_params, scores)[0, 1]
        
        # Fit line in log space
        z = np.polyfit(log_params, scores, 1)
        p = np.poly1d(z)
        
        # Plot trend line
        x_range = np.logspace(np.min(log_params), np.max(log_params), 100)
        plt.plot(x_range, p(np.log10(x_range)), 
                "k--", alpha=0.5, label='Trend line')
        
        # Add labels
        texts = []
        for data in plot_data:
            model_name = data['model'].split('/')[-1]
            texts.append(plt.text(data['parameters'], data['score'], 
                                model_name, fontsize=8))
        
        # Adjust text positions with transformed coordinates
        adjustText.adjust_text(texts, 
                             arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5),
                             force_points=(0.1, 0.1),
                             expand_points=(1.5, 1.5))
        
        # Add correlation text
        plt.text(0.05, 0.95, f'Log-Scale Correlation: {correlation:.3f}',
                transform=plt.gca().transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Customize plot
        plt.xlabel('Number of Parameters (log scale)')
        plt.ylabel(f'AidanBench {metric_name}')
        plt.title(f'Model Size vs {metric_name}')
        
        # Format x-axis tick labels
        ax = plt.gca()
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
        
        # Add legend for companies
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=color, label=company, markersize=10)
                         for company, color in COMPANY_COLORS.items()]
        legend_elements.append(plt.Line2D([0], [0], linestyle='--', color='k', alpha=0.5,
                                        label='Trend line'))
        plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
        
        # Add grid that respects log scale
        plt.grid(True, alpha=0.3, which='both')
        
        # Save plot
        plt.tight_layout()
        plt.savefig(Path(output_dir) / f'parameter_comparison_{metric_key}.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()



def analyze_parameter_scaling(results_file: str,
                            model_scales: List[dict],
                            output_dir: str = 'plots') -> None:
    """Main function to analyze performance versus model size."""
    
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate comparison plots
    create_parameter_plots(results, model_scales, output_dir)



#####

def create_comprehensive_table(results: dict,
                             output_dir: str = 'tables',
                             min_embedding_threshold: float = 0.15,
                             min_coherence_threshold: float = 15.0) -> None:
    """
    Create CSV tables showing all scores for each question and model.
    Creates three tables: one each for embedding, coherence, and valid answers.
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get all models and questions
    models = sorted(results['models'].keys())
    questions = set()
    for model_data in results['models'].values():
        for temp_data in model_data.values():
            questions.update(temp_data.keys())
    questions = sorted(questions)
    
    # Initialize score storage
    scores = {
        'embedding': defaultdict(lambda: defaultdict(float)),
        'coherence': defaultdict(lambda: defaultdict(float)),
        'answers': defaultdict(lambda: defaultdict(int))
    }
    
    # Calculate scores for each model-question pair
    for model in models:
        model_data = results['models'][model]
        for temp, questions_data in model_data.items():
            for question, answers in questions_data.items():
                embedding_total = 0
                coherence_total = 0
                valid_answers = 0
                
                for answer in answers:
                    if (answer['embedding_dissimilarity_score'] >= min_embedding_threshold and 
                        answer['coherence_score'] >= min_coherence_threshold):
                        embedding_total += answer['embedding_dissimilarity_score']
                        coherence_total += answer['coherence_score'] / 100  # Divide by 100 as requested
                        valid_answers += 1
                
                # Update maximum scores for this model-question pair
                scores['embedding'][question][model] = max(
                    scores['embedding'][question][model],
                    embedding_total
                )
                scores['coherence'][question][model] = max(
                    scores['coherence'][question][model],
                    coherence_total
                )
                scores['answers'][question][model] = max(
                    scores['answers'][question][model],
                    valid_answers
                )
    
    # Create CSV files
    for metric in ['embedding', 'coherence', 'answers']:
        filename = Path(output_dir) / f'question_scores_{metric}.csv'
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            header = ['Question'] + [model.split('/')[-1] for model in models]
            writer.writerow(header)
            
            # Write data
            for question in questions:
                row = [question]
                for model in models:
                    score = scores[metric][question][model]
                    if metric in ['embedding', 'coherence']:
                        row.append(f'{score:.3f}')
                    else:
                        row.append(str(score))
                writer.writerow(row)
    
    # Also create a combined Excel file with all metrics
    # try:
    if True:
        # import pandas as pd
        
        # Create DataFrames for each metric
        dfs = {}
        for metric in ['embedding', 'coherence', 'answers']:
            data = []
            for question in questions:
                row = {'Question': question}
                for model in models:
                    score = scores[metric][question][model]
                    if metric in ['embedding', 'coherence']:
                        row[model.split('/')[-1]] = f'{score:.3f}'
                    else:
                        row[model.split('/')[-1]] = str(score)
                data.append(row)
            dfs[metric] = pd.DataFrame(data)
        
        # Save to Excel with multiple sheets
        with pd.ExcelWriter(Path(output_dir) / 'question_scores_all.xlsx') as writer:
            for metric, df in dfs.items():
                df.to_excel(writer, sheet_name=metric, index=False)
                
    # except ImportError:
    #     print("pandas not installed - skipping Excel file creation")

def generate_score_tables(results_file: str,
                         output_dir: str = 'tables') -> None:
    """Main function to generate score tables."""
    
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Generate tables
    create_comprehensive_table(results, output_dir)



#####


import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

def calculate_model_scores_with_threshold(results: dict,
                                        embedding_threshold: float,
                                        coherence_threshold: float) -> Dict[str, Dict[str, float]]:
    """
    Calculate total scores for each model with given thresholds.
    Stop counting at first answer below threshold.
    """
    model_scores = defaultdict(lambda: {'embedding': 0, 'coherence': 0, 'answers': 0})
    
    for model_name, temp_data in results['models'].items():
        # Track best scores across temperatures
        best_scores = defaultdict(float)
        
        for temp, questions in temp_data.items():
            temp_embedding = 0
            temp_coherence = 0
            temp_answers = 0
            
            for question, answers in questions.items():
                # Only look at answers until we hit one below threshold
                for answer in answers:
                    if (answer['embedding_dissimilarity_score'] >= embedding_threshold and 
                        answer['coherence_score'] >= coherence_threshold):
                        temp_embedding += answer['embedding_dissimilarity_score']
                        temp_coherence += answer['coherence_score'] / 100
                        temp_answers += 1
                    else:
                        break  # Stop at first answer below threshold
            
            # Update best scores for this model
            best_scores['embedding'] = max(best_scores['embedding'], temp_embedding)
            best_scores['coherence'] = max(best_scores['coherence'], temp_coherence)
            best_scores['answers'] = max(best_scores['answers'], temp_answers)
        
        model_scores[model_name] = dict(best_scores)
    
    return model_scores

def create_threshold_tables(results: dict, output_dir: str = 'tables') -> None:
    """Create tables showing score and ranking evolution with different thresholds."""
    
    # Generate threshold values
    embedding_thresholds = np.arange(0.15, 1.0, 0.1)
    coherence_thresholds = np.arange(15, 100, 10)
    
    # Store scores and rankings for each threshold
    scores_data = []
    rankings_data = []
    
    for emb_thresh, coh_thresh in zip(embedding_thresholds, coherence_thresholds):
        # Calculate scores for this threshold
        scores = calculate_model_scores_with_threshold(results, emb_thresh, coh_thresh)
        
        # Prepare score data
        for model, metrics in scores.items():
            scores_data.append({
                'Embedding Threshold': f'{emb_thresh:.2f}',
                'Coherence Threshold': f'{coh_thresh:.0f}',
                'Model': model.split('/')[-1],
                'Company': model.split('/')[0],
                'Embedding Score': metrics['embedding'],
                'Coherence Score': metrics['coherence'],
                'Valid Answers': metrics['answers']
            })
        
        # Calculate rankings
        models = list(scores.keys())
        for metric in ['embedding', 'coherence', 'answers']:
            sorted_models = sorted(models, 
                                 key=lambda m: scores[m][metric],
                                 reverse=True)
            for rank, model in enumerate(sorted_models, 1):
                rankings_data.append({
                    'Embedding Threshold': f'{emb_thresh:.2f}',
                    'Coherence Threshold': f'{coh_thresh:.0f}',
                    'Model': model.split('/')[-1],
                    'Company': model.split('/')[0],
                    'Metric': metric,
                    'Rank': rank
                })
    
    # Create DataFrames
    scores_df = pd.DataFrame(scores_data)
    rankings_df = pd.DataFrame(rankings_data)
    
    # Create threshold labels
    threshold_labels = [f'E:{e:.2f}/C:{c:.0f}' 
                       for e, c in zip(embedding_thresholds, coherence_thresholds)]
    
    # Create more readable pivot tables
    # Scores tables (one for each metric)
    scores_tables = {}
    for metric in ['Embedding Score', 'Coherence Score', 'Valid Answers']:
        pivot = pd.pivot_table(
            scores_df,
            index=['Company', 'Model'],
            columns=['Embedding Threshold', 'Coherence Threshold'],
            values=metric,
            aggfunc='first'
        ).round(3)
        
        # Flatten column index and rename with more readable labels
        pivot.columns = threshold_labels
        scores_tables[metric] = pivot
    
    # Rankings table
    rankings_pivot = pd.pivot_table(
        rankings_df,
        index=['Company', 'Model', 'Metric'],
        columns=['Embedding Threshold', 'Coherence Threshold'],
        values='Rank',
        aggfunc='first'
    )
    rankings_pivot.columns = threshold_labels
    
    # Save to Excel
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(Path(output_dir) / 'threshold_analysis.xlsx') as writer:
        for metric, table in scores_tables.items():
            sheet_name = metric.replace(' ', '_')[:31]  # Excel sheet name length limit
            table.to_excel(writer, sheet_name=sheet_name)
        rankings_pivot.to_excel(writer, sheet_name='Rankings')
    
    # Save CSVs
    for metric, table in scores_tables.items():
        filename = f'threshold_scores_{metric.lower().replace(" ", "_")}.csv'
        table.to_csv(Path(output_dir) / filename)
    rankings_pivot.to_csv(Path(output_dir) / 'threshold_rankings.csv')

def analyze_thresholds(results_file: str, output_dir: str = 'tables') -> None:
    """Main function to analyze threshold sensitivity."""
    
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Generate tables
    create_threshold_tables(results, output_dir)

if __name__ == "__main__":
    analyze_thresholds('results.json', output_dir='tables')