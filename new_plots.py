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
from model_list import lmsys_scores, release_dates, model_scales, model_prices

from scipy import stats
import adjustText

from datetime import datetime, timedelta
import matplotlib.dates as mdates

import csv
import pandas as pd
import textwrap


# Define company colors
COMPANY_COLORS = {
    'openai': '#74AA9C',      # OpenAI green
    'meta-llama': '#044EAB',  # Meta blue
    'anthropic': '#D4C5B9',   # Anthropic beige
    'google': '#669DF7',      # Google blue
    'x-ai': '#000000',        # X black
    'mistralai': '#F54E42'    # Mistral red
}

# Add these constants at the top with other constants
PAPER_FONT_SIZE = 12  # Base font size for paper
PAPER_TITLE_SIZE = 14  # Title font size
PAPER_LABEL_SIZE = 10  # Label font size for bars/points
FIGURE_WIDTH = 10  # Standard figure width for paper
FIGURE_HEIGHT = 6  # Base figure height (will be adjusted based on content)

def _set_paper_style():
    """Configure plot style for academic paper presentation."""
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        'font.size': PAPER_FONT_SIZE,
        'axes.labelsize': PAPER_FONT_SIZE,
        'axes.titlesize': PAPER_TITLE_SIZE,
        'xtick.labelsize': PAPER_FONT_SIZE,
        'ytick.labelsize': PAPER_FONT_SIZE,
        'legend.fontsize': PAPER_FONT_SIZE,
        'figure.titlesize': PAPER_TITLE_SIZE,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1
    })

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
    """Create a vertical bar plot for the specified metric, sorted by value."""
    _set_paper_style()
    
    models = list(metrics.keys())
    temps = list(metrics[models[0]].keys())
    
    # Calculate max value for each model for ranking
    max_values = {model: max(metric_getter(temp_metrics) 
                           for temp_metrics in temp_data.values())
                 for model, temp_data in metrics.items()}
    
    # Sort models by their max values
    sorted_models = sorted(models, key=lambda x: max_values[x], reverse=True)
    
    # Set up the plot with adjusted dimensions
    plt.figure(figsize=(FIGURE_WIDTH, max(FIGURE_HEIGHT * 1.2, len(models) * 0.4)))
    
    # Calculate bar positions
    bar_width = 0.8 / len(temps)
    positions_base = np.arange(len(models))
    
    # Create bars for each temperature
    for i, temp in enumerate(temps):
        positions = positions_base + i * bar_width
        values = [metric_getter(metrics[model][temp]) for model in sorted_models]
        
        if color_by_company:
            # Create bars with company-specific colors
            for pos, val, model in zip(positions, values, sorted_models):
                company = model.split('/')[0]
                bar = plt.bar(pos, val, bar_width, 
                            color=COMPANY_COLORS[company],
                            alpha=0.8)
                # Add value label
                plt.text(pos, val + max(values) * 0.01,  # Slight offset above bar
                        format_value(val),
                        ha='center',
                        va='bottom',
                        rotation=45,
                        fontsize=PAPER_LABEL_SIZE)
        else:
            # Create bars with default coloring
            container = plt.bar(positions, values, bar_width, 
                              label=f'Temperature {temp}' if not paper_ready else None,
                              alpha=0.8)
            # Add value labels
            for rect, val in zip(container, values):
                plt.text(rect.get_x() + rect.get_width()/2, val + max(values) * 0.01,
                        format_value(val),
                        ha='center',
                        va='bottom',
                        rotation=45,
                        fontsize=PAPER_LABEL_SIZE)
    
    # Customize the plot
    plt.xlabel('Models')
    plt.ylabel(ylabel)
    if paper_ready:
        plt.title(f'{metric_name} by Model')
    else:
        plt.title(f'{metric_name} by Model and Temperature\n(Filtered for embedding ≥ 0.15 and coherence ≥ 15)')
    
    # Position model names
    plt.xticks(positions_base + (bar_width * (len(temps) - 1)) / 2, 
               [m.split('/')[-1] for m in sorted_models],
               rotation=45,
               ha='right')
    
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
    
    # Add grid and adjust layout
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    # Ensure there's enough space at the top for labels
    plt.margins(y=0.15)
    
    # Save the plot
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
    
    # Add this line to generate the new results plot
    plot_results(metrics, output_dir)
    
    # Rest of your existing plot generations...
    plot_metric(
        metrics,
        "Total Embedding Dissimilarity",
        lambda x: x.embedding_total,
        output_dir,
        'embedding_scores.png',
        'Total Embedding Dissimilarity Score',
        lambda x: f"{x:.2f}"
    )
    
    # ... (rest of the function remains the same)


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

def _create_styled_label(ax, x, y, text: str, color: str, offset_x: float = 0.005) -> plt.Text:
    """Create a consistently styled text label with background."""
    # Calculate offset based on axis scale type
    if isinstance(x, datetime):
        date_range = (ax.get_xlim()[1] - ax.get_xlim()[0])
        x_offset = timedelta(days=date_range * offset_x)
    else:
        x_offset = x * 0.005 if x > 0 else 0.005
    
    # Create ultra-soft background color with same alpha as scatter points
    rgba_color = plt.matplotlib.colors.to_rgba(color, alpha=0.2)  # Match scatter point alpha
    
    text_obj = ax.text(x + x_offset, y, text,
                      fontsize=11,
                      bbox=dict(
                          facecolor=rgba_color,
                          alpha=0.3,  # Reduced transparency for label background
                          edgecolor='none',
                          boxstyle='round,pad=0.4,rounding_size=0.4',
                          linewidth=0
                      ),
                      horizontalalignment='left',
                      verticalalignment='center',
                      zorder=100)
    
    text_obj.point_coords = (x, y)
    return text_obj

def _adjust_labels(texts, ax):
    """Helper function to adjust label positions using adjustText with minimal movement."""
    adjustText.adjust_text(
        texts,
        ax=ax,
        force_points=(0.01, 0.01),   # Keep minimal force between points and texts
        force_text=(0.05, 0.05),     # Keep very small force between texts
        expand_points=(0.5, 0.5),    # Keep minimal point expansion
        expand_text=(0.5, 0.5),      # Keep minimal text expansion
        arrowprops=dict(
            arrowstyle='-',
            color='gray',
            alpha=0.5,              # Restored original alpha
            lw=0.5                  # Restored original line width
        ),
        avoid_self=True,
        avoid_points=True,
        min_arrow_dist=1.0,        # Keep minimal arrow distance
        autoalign=False,
        only_move={'points': 'xy',
                  'text': 'xy',
                  'objects': 'xy'},
        text_from_point=(0, 0),
        save_steps=False,
        drag_enable=False
    )

def create_timeline_plots(results: dict,
                        release_dates: List[dict],
                        output_dir: str = 'plots') -> None:
    """
    Create scatter plots showing score evolution over time with log scale.
    """
    # Get maximum scores for each model
    max_scores = get_max_scores(results)
    
    # Convert release dates to dict and datetime objects
    release_dict = {item['model']: datetime.strptime(item['release_date'], '%Y-%m-%d')
                   for item in release_dates}
    
    # Create scatter plot
    plt.figure(figsize=(15, 8))
    
    # Collect data points
    plot_data = []
    for model, scores in max_scores.items():
        if model in release_dict:
            plot_data.append({
                'model': model,
                'answers': scores['answers'],  # Focus on number of answers
                'date': release_dict[model],
                'company': model.split('/')[0]
            })
    
    # Get date range
    dates = [d['date'] for d in plot_data]
    min_date = min(dates)
    max_date = max(dates)
    
    # Add padding to date range (10% on each side)
    date_range = (max_date - min_date).days
    padding = timedelta(days=int(date_range * 0.1))
    plt.xlim(min_date - padding, max_date + padding)
    
    # Plot points with fill color
    for data in plot_data:
        plt.scatter(data['date'], data['answers'],
                   color=COMPANY_COLORS[data['company']],
                   alpha=0.2,  # Match label background alpha
                   s=100,
                   edgecolor=COMPANY_COLORS[data['company']],
                   linewidth=1,
                   zorder=10)
    
    # Add labels with background - directly next to points
    for data in plot_data:
        model_name = data['model'].split('/')[-1]
        _create_styled_label(plt.gca(), 
                           data['date'], 
                           data['answers'], 
                           model_name, 
                           COMPANY_COLORS[data['company']])
    
    # Set log scale for y-axis
    plt.yscale('log')
    
    # Customize plot
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    
    # Rotate and align the tick labels so they look better
    plt.gcf().autofmt_xdate()
    
    plt.xlabel('Release Date')
    plt.ylabel('Number of Valid Answers (log scale)')
    plt.title('Model Performance Evolution Over Time')
    
    # Add legend for companies
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                markerfacecolor=color, label=company, markersize=10)
                     for company, color in COMPANY_COLORS.items()]
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'model_timeline.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_parameter_plots(results: dict,
                         model_scales: List[dict],
                         output_dir: str = 'plots') -> None:
    """Create scatter plots comparing performance versus parameter count for Llama models."""
    max_scores = get_max_scores(results)
    scale_dict = {item['model']: item['parameters'] for item in model_scales if item['model'].startswith("meta")}
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Collect and sort data points by parameter count
    plot_data = []
    for model, scores in max_scores.items():
        if model in scale_dict:
            plot_data.append({
                'model': model,
                'answers': scores['answers'],
                'parameters': scale_dict[model],
                'company': model.split('/')[0]
            })
    
    # Sort by parameter count to help with label placement
    plot_data.sort(key=lambda x: x['parameters'])
    
    # Plot points
    for data in plot_data:
        ax.scatter(data['parameters'], data['answers'],
                  color=COMPANY_COLORS[data['company']],
                  alpha=0.2,  # Match label background alpha
                  s=100,
                  edgecolor=COMPANY_COLORS[data['company']],
                  linewidth=1,
                  zorder=10)
    
    # Set scales before adding labels
    ax.set_xscale('log')
    # ax.set_yscale('log')
    
    # Add labels with adjusted positions
    texts = []
    for data in plot_data:
        model_name = data['model'].split('/')[-1]
        texts.append(_create_styled_label(ax,
                                        data['parameters'],
                                        data['answers'],
                                        model_name,
                                        COMPANY_COLORS[data['company']]))
    
    # Adjust labels with optimized parameters
    _adjust_labels(texts, ax)
    
    # Calculate correlation with log values
    log_params = np.log10([d['parameters'] for d in plot_data])
    not_log_answers = [d['answers'] for d in plot_data]
    correlation = np.corrcoef(log_params, not_log_answers)[0, 1]
    
    # Add correlation text
    plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
            transform=plt.gca().transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.xlabel('Number of Parameters (log scale)')
    plt.ylabel('Number of Valid Answers')
    plt.title('Llama Family Performance by Model Size')
    
    # Format x-axis tick labels
    ax = plt.gca()
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    # Add legend for companies
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                markerfacecolor=color, label=company, markersize=10)
                     for company, color in COMPANY_COLORS.items()]
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Add grid that respects log scale
    plt.grid(True, alpha=0.3, which='both')
    
    # Save plot
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'llama_size_performance.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_cost_performance_plots(results: dict,
                                model_prices: List[dict],
                                output_dir: str = 'plots') -> None:
    """Create scatter plots comparing model performance versus cost."""
    max_scores = get_max_scores(results)
    cost_dict = {item['model']: (item['input_price'] + item['output_price'])/2 
                 for item in model_prices}
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Collect and sort data points by cost
    plot_data = []
    for model, scores in max_scores.items():
        if model in cost_dict:
            plot_data.append({
                'model': model,
                'answers': scores['answers'],
                'cost': cost_dict[model],
                'company': model.split('/')[0]
            })
    
    # Sort by cost to help with label placement
    plot_data.sort(key=lambda x: x['cost'])
    
    # Plot points
    for data in plot_data:
        ax.scatter(data['cost'], data['answers'],
                  color=COMPANY_COLORS[data['company']],
                  alpha=0.2,  # Match label background alpha
                  s=100,
                  edgecolor=COMPANY_COLORS[data['company']],
                  linewidth=1,
                  zorder=10)
    
    # Set scales before adding labels
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Add labels with adjusted positions
    texts = []
    for data in plot_data:
        model_name = data['model'].split('/')[-1]
        texts.append(_create_styled_label(ax,
                                        data['cost'],
                                        data['answers'],
                                        model_name,
                                        COMPANY_COLORS[data['company']]))
    
    # Adjust labels with optimized parameters
    _adjust_labels(texts, ax)
    
    # Calculate correlation with log values
    log_costs = np.log10([d['cost'] for d in plot_data])
    log_answers = np.log10([d['answers'] for d in plot_data])
    correlation = np.corrcoef(log_costs, log_answers)[0, 1]
    
    # Add correlation text
    plt.text(0.05, 0.95, f'Log-Scale Correlation: {correlation:.3f}',
            transform=plt.gca().transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.xlabel('Average Cost per 1K Tokens (USD, log scale)')
    plt.ylabel('Number of Valid Answers (log scale)')
    plt.title('Cost vs Performance')
    
    # Add legend for companies
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                markerfacecolor=color, label=company, markersize=10)
                     for company, color in COMPANY_COLORS.items()]
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Add grid that respects log scale
    plt.grid(True, alpha=0.3, which='both')
    
    # Save plot
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'cost_performance.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

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
    

def analyze_model_performance(results_file: str,
                            release_dates: List[dict],
                            model_scales: List[dict],
                            model_prices: List[dict],
                            lmsys_scores: List[dict],
                            output_dir: str = 'plots') -> None:
    """Generate all performance analysis plots."""
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Calculate metrics for the results plot
    metrics = calculate_metrics(
        results,
        min_embedding_threshold=0.15,
        min_coherence_threshold=15.0
    )
    
    # Generate the results plot
    plot_results(metrics, output_dir)
    
    # Generate all other plots
    create_timeline_plots(results, release_dates, output_dir)
    create_timeline_progression(results, release_dates, output_dir)
    create_parameter_plots(results, model_scales, output_dir)
    create_cost_performance_plots(results, model_prices, output_dir)
    plot_lmsys_correlation(results, lmsys_scores, output_dir)
    plot_exit_reasons(results, model_scales, output_dir)
    plot_top_questions(results, output_dir)
    
    # Generate tables
    generate_score_tables(results_file, 'tables')
    

def plot_lmsys_correlation(results: dict,
                          lmsys_scores: List[dict],
                          output_dir: str = 'plots') -> None:
    """Create scatter plot comparing model performance vs LMSYS scores with log scaling."""
    fig, ax = plt.subplots(figsize=(15, 8))  # Match size with other plots
    
    # Calculate max scores for each model
    max_scores = get_max_scores(results)
    
    # Create score mapping
    lmsys_map = {item['model']: item['lmsys_score'] for item in lmsys_scores}
    
    # Collect and sort data points
    plot_data = []
    for model, scores in max_scores.items():
        if model in lmsys_map:
            plot_data.append({
                'model': model,
                'answers': scores['answers'],
                'lmsys_score': lmsys_map[model],
                'company': model.split('/')[0]
            })
    
    # Sort by LMSYS score to help with label placement
    plot_data.sort(key=lambda x: x['lmsys_score'])
    
    # Plot points
    for data in plot_data:
        ax.scatter(data['lmsys_score'], data['answers'],
                  color=COMPANY_COLORS[data['company']],
                  alpha=0.2,  # Match label background alpha
                  s=100,
                  edgecolor=COMPANY_COLORS[data['company']],
                  linewidth=1,
                  zorder=10)
    
    # Set scales before adding labels
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    
    # Add labels with adjusted positions
    texts = []
    for data in plot_data:
        model_name = data['model'].split('/')[-1]
        texts.append(_create_styled_label(ax,
                                        data['lmsys_score'],
                                        data['answers'],
                                        model_name,
                                        COMPANY_COLORS[data['company']]))
    
    # Adjust labels with optimized parameters
    _adjust_labels(texts, ax)
    
    # Calculate correlation with log values
    not_log_lmsys = [d['lmsys_score'] for d in plot_data]
    not_log_answers = [d['answers'] for d in plot_data]
    correlation = np.corrcoef(not_log_lmsys, not_log_answers)[0, 1]
    
    # Add correlation text in same style as other plots
    plt.text(0.05, 0.95, f'Log-Scale Correlation: {correlation:.3f}',
            transform=plt.gca().transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.xlabel('LMSYS Score')
    plt.ylabel('Number of Valid Answers')
    plt.title('LMSYS Score vs Performance')
    
    # Add legend for companies in same style as other plots
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                markerfacecolor=color, label=company, markersize=10)
                     for company, color in COMPANY_COLORS.items()]
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Add grid that respects log scale
    plt.grid(True, alpha=0.3, which='both')
    
    # Save plot
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'lmsys_correlation.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_exit_reasons(results: dict,
                     model_scales: List[dict],
                     output_dir: str = 'plots') -> None:
    """Create a stacked bar chart showing the composition of exit reasons by model."""
    _set_paper_style()
    
    # Collect exit reason data
    model_stats = defaultdict(lambda: {'coherence': 0, 'novelty': 0, 'total': 0})
    scale_dict = {item['model']: item['parameters'] for item in model_scales}
    
    for model, temp_data in results['models'].items():
        for temp, questions in temp_data.items():
            for answers in questions.values():
                model_stats[model]['total'] += 1
                
                # Track if coherence break occurred first
                coherence_break = False
                for answer in answers:
                    if answer['coherence_score'] < 15:
                        coherence_break = True
                        break
                
                if coherence_break:
                    model_stats[model]['coherence'] += 1
                else:
                    # All other cases (including no break) count as novelty breaks
                    model_stats[model]['novelty'] += 1
    
    # Convert to percentages and prepare data for plotting
    models = []
    coherence_pcts = []
    novelty_pcts = []
    parameters = []
    
    for model, stats in model_stats.items():
        total = stats['total']
        if total > 0:  # Only include models with data
            models.append(model)
            coherence_pcts.append((stats['coherence'] / total) * 100)
            novelty_pcts.append((stats['novelty'] / total) * 100)
            parameters.append(scale_dict.get(model, 0))
    
    # Sort by novelty percentage (descending)
    sort_idx = np.argsort(novelty_pcts)[::-1]
    
    models = [models[i] for i in sort_idx]
    coherence_pcts = [coherence_pcts[i] for i in sort_idx]
    novelty_pcts = [novelty_pcts[i] for i in sort_idx]
    parameters = [parameters[i] for i in sort_idx]
    
    # Create figure with extra space for labels
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0])
    
    # Plot stacked bars
    bar_width = 0.8
    x = np.arange(len(models))
    
    colors = {
        'coherence': '#FF6B6B',  # Warm coral red for coherence breaks
        'novelty': '#4ECDC4'     # Cool turquoise for novelty breaks
    }
    
    # Plot each category
    coherence_bars = ax.bar(x, coherence_pcts, bar_width,
                          label='Coherence Break', color=colors['coherence'])
    
    novelty_bars = ax.bar(x, novelty_pcts, bar_width,
                       label='Novelty Break', color=colors['novelty'],
                       bottom=coherence_pcts)
    
    # Add percentage labels on bars
    def add_labels(bars, percentages, bottom_vals):
        for idx, (bar, pct) in enumerate(zip(bars, percentages)):
            if pct > 5:  # Only show label if segment is large enough
                height = bar.get_height()
                y_pos = bottom_vals[idx] + height/2
                ax.text(idx, y_pos, f'{pct:.1f}%',
                       ha='center', va='center',
                       fontsize=8)
    
    # Add labels for each segment
    add_labels(coherence_bars, coherence_pcts, np.zeros(len(models)))
    add_labels(novelty_bars, novelty_pcts, coherence_pcts)
    
    # Customize plot
    ax.set_xticks(x)
    ax.set_xticklabels([m.split('/')[-1] for m in models],
                       rotation=45, ha='right')
    ax.set_ylabel('Percentage of Sequences')
    ax.set_title('Exit Reason Distribution by Model\n(Sorted by Novelty Break Percentage)')
    
    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add parameter count as secondary x-axis if available
    if any(parameters):
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'{p:,}' if p > 0 else 'N/A' for p in parameters],
                           rotation=45, ha='left')
        ax2.set_xlabel('Parameter Count')
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='y')
    
    # Adjust layout and save
    plt.subplots_adjust(right=0.85, bottom=0.2)
    plt.savefig(Path(output_dir) / 'exit_reasons.png',
                dpi=300, bbox_inches='tight')
    plt.close()

def create_timeline_progression(results: dict,
                              release_dates: List[dict],
                              output_dir: str = 'plots') -> None:
    """
    Create a timeline plot showing performance progression with colored arrows 
    connecting best models for each company over time.
    """
    # Get maximum scores for each model
    max_scores = get_max_scores(results)
    
    # Convert release dates to dict and datetime objects
    release_dict = {item['model']: datetime.strptime(item['release_date'], '%Y-%m-%d')
                   for item in release_dates}
    
    # Collect and organize data by company
    company_data = defaultdict(list)
    for model, scores in max_scores.items():
        if model in release_dict:
            company = model.split('/')[0]
            company_data[company].append({
                'model': model,
                'answers': scores['answers'],
                'date': release_dict[model]
            })
    
    # Sort each company's models by date
    for company in company_data:
        company_data[company].sort(key=lambda x: x['date'])
    
    # Create the plot
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    
    # Get date range
    all_dates = [d['date'] for company_models in company_data.values() 
                for d in company_models]
    min_date = min(all_dates)
    max_date = max(all_dates)
    date_range = (max_date - min_date).days
    padding = timedelta(days=int(date_range * 0.1))
    
    # Plot points and arrows for each company
    for company, models in company_data.items():
        color = COMPANY_COLORS[company]
        
        # Plot points
        dates = [m['date'] for m in models]
        scores = [m['answers'] for m in models]
        
        plt.scatter(dates, scores,
                   color=color,
                   alpha=0.2,
                   s=100,
                   edgecolor=color,
                   linewidth=1,
                   zorder=10)
        
        # Add arrows connecting points
        if len(models) > 1:
            for i in range(len(models) - 1):
                dx = (models[i+1]['date'] - models[i]['date']).days
                dy = models[i+1]['answers'] - models[i]['answers']
                
                plt.arrow(models[i]['date'],
                         models[i]['answers'],
                         dx,
                         dy,
                         color=color,
                         alpha=0.6,
                         length_includes_head=True,
                         head_width=max(models[i]['answers'] * 0.05, 1),
                         head_length=date_range * 0.02,
                         zorder=5)
        
        # Add labels
        for model in models:
            model_name = model['model'].split('/')[-1]
            _create_styled_label(ax,
                               model['date'],
                               model['answers'],
                               model_name,
                               color)
    
    # Set log scale for y-axis
    plt.yscale('log')
    
    # Customize plot
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gcf().autofmt_xdate()
    
    plt.xlim(min_date - padding, max_date + padding)
    
    plt.xlabel('Release Date')
    plt.ylabel('Number of Valid Answers (log scale)')
    plt.title('Model Performance Evolution Over Time\nwith Company Progression Paths')
    
    # Add legend for companies
    legend_elements = [plt.Line2D([0], [0], marker='o', color=color, 
                                label=company, markersize=10,
                                alpha=0.8)
                     for company, color in COMPANY_COLORS.items()]
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'model_timeline_progression.png',
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_top_questions(results: dict, output_dir: str = 'plots', n_questions: int = 7) -> None:
    """Create a violin plot showing answer distribution for top N most answered questions."""
    _set_paper_style()
    
    # Calculate total answers per question
    question_totals = defaultdict(int)
    for model_data in results['models'].values():
        for temp_data in model_data.values():
            for question, answers in temp_data.items():
                question_totals[question] += len(answers)
    
    # Get top N questions
    top_questions = sorted(question_totals.items(), key=lambda x: x[1], reverse=True)[:n_questions]
    
    # Collect answer counts by company for each question
    plot_data = []
    for question, _ in top_questions:
        for model, temp_data in results['models'].items():
            company = model.split('/')[0]
            for temp, questions in temp_data.items():
                if question in questions:
                    plot_data.append({
                        'Question': question,
                        'Company': company,
                        'Answers': len(questions[question])
                    })
    
    # Convert to DataFrame
    df = pd.DataFrame(plot_data)
    
    # Create figure with more compact size
    plt.figure(figsize=(8, 5))  # Reduced from (12, 6)
    
    # Create violin plot
    sns.violinplot(data=df,
                  x='Question',
                  y='Answers',
                  hue='Company',
                  palette=COMPANY_COLORS,
                  inner='points',
                  scale='width',
                  cut=0)
    
    # Customize plot
    plt.xticks(rotation=45, ha='right')
    
    # Wrap question text more aggressively
    ax = plt.gca()
    labels = ax.get_xticklabels()
    wrapped_labels = [textwrap.fill(label.get_text(), width=20) for label in labels]  # Reduced width from 30
    ax.set_xticklabels(wrapped_labels, fontsize=8)  # Reduced font size
    
    # Move legend to the top to save horizontal space
    plt.legend(bbox_to_anchor=(0.5, 1.15), 
              loc='upper center', 
              borderaxespad=0,
              ncol=3,  # Arrange legend in 3 columns
              fontsize=8)  # Reduced font size
    
    # Add title with padding
    plt.title('Answer Distribution - Top 7 Most Answered Questions', 
             pad=30,  # Increased padding for legend
             fontsize=10)  # Reduced font size
    
    # Adjust margins
    plt.subplots_adjust(top=0.85,  # Make room for legend
                       bottom=0.25,  # Make room for labels
                       right=0.95,
                       left=0.1)
    
    # Save plot with tighter margins
    plt.savefig(Path(output_dir) / 'top_questions_distribution.png',
                dpi=300,
                bbox_inches='tight',
                pad_inches=0.2)  # Reduced padding
    plt.close()

def plot_results(metrics: Dict[str, Dict[str, ModelMetrics]], 
                output_dir: str = 'plots') -> None:
    """Create a simple horizontal bar chart showing overall model results, highest at top."""
    _set_paper_style()
    
    # Calculate total valid answers for each model
    model_scores = {model: max(metrics[model][temp].valid_answers 
                             for temp in metrics[model].keys())
                   for model in metrics.keys()}
    
    # Sort models by score
    sorted_items = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
    models, values = zip(*sorted_items)
    
    # Set up the plot with swapped dimensions for horizontal layout
    plt.figure(figsize=(FIGURE_WIDTH, max(FIGURE_HEIGHT, len(models) * 0.5)))
    
    # Create horizontal bars with company colors
    bars = []
    positions = range(len(models))[::-1]  # Reverse the positions to put highest at top
    for i, (model, value) in enumerate(zip(models, values)):
        company = model.split('/')[0]
        bar = plt.barh(positions[i], value, 
                      color=COMPANY_COLORS[company],
                      alpha=0.8)
        bars.append(bar)
        
        # Add value label to the right of each bar
        plt.text(value + max(values) * 0.01,  # Slight offset from bar end
                positions[i],
                str(int(value)),
                va='center',
                fontsize=PAPER_LABEL_SIZE)
    
    # Customize the plot
    plt.xlabel('Number of Valid Responses')
    plt.title('Model Results')
    
    # Position model names on y-axis
    plt.yticks(positions, 
               [m.split('/')[-1] for m in models])
    
    # Add company color legend
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, label=company) 
                      for company, color in COMPANY_COLORS.items()]
    plt.legend(handles=legend_elements,
              bbox_to_anchor=(1.05, 1),
              loc='upper left')
    
    # Add grid and adjust layout
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    
    # Ensure there's enough space at the right for labels
    plt.margins(x=0.15)
    
    # Save the plot
    plt.savefig(Path(output_dir) / 'results.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    from model_list import release_dates, model_scales, model_prices, lmsys_scores
    
    # Run the analysis with LMSYS scores
    analyze_model_performance(
        results_file='results.json',
        release_dates=release_dates,
        model_scales=model_scales,
        model_prices=model_prices,
        lmsys_scores=lmsys_scores,
        output_dir='plots'
    )
    
