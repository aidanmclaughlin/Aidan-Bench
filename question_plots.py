import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from scipy import stats
from scipy.stats import t
import statsmodels.stats.api as sms
from typing import Tuple

COMPANY_COLORS = {
    'openai': '#74AA9C',
    'meta': '#044EAB',
    'anthropic': '#D4C5B9',
    'google': '#669DF7',
    'x': '#000000',
    'mistral': '#F54E42'
}


def _get_company_from_model(model_name: str) -> str:
    company = model_name.split('/')[0]
    if company == 'meta-llama':
        return 'meta'
    if company == 'x-ai':
        return 'x'
    if company == 'mistralai':
        return 'mistral'
    return company


def _load_results() -> dict:
    with open('results.json', 'r') as f:
        return json.load(f)


def _extract_scores(results: dict) -> pd.DataFrame:
    data = []
    for model, model_data in results['models'].items():
        for temperature, temp_data in model_data.items():
            for question, answers in temp_data.items():
                for answer in answers:
                    data.append({
                        'model': model,
                        'company': _get_company_from_model(model),
                        'temperature': float(temperature),
                        'question': question,
                        'answer_num': answer['answer_num'],
                        'embedding_score': answer['embedding_dissimilarity_score'],
                        'coherence_score': answer['coherence_score']
                    })
    return pd.DataFrame(data)


def _get_best_models(df: pd.DataFrame) -> dict:
    # Hardcoded best models based on latest scores
    best_models = {
        'embedding_score': {
            'openai': 'openai/o1-preview',
            'anthropic': 'anthropic/claude-3.5-sonnet',
            'google': 'google/gemini-pro-1.5',
            'meta': 'meta-llama/llama-3.1-405b-instruct',
            'mistral': 'mistralai/mistral-large-latest',
            'x': 'x-ai/grok-beta'
        },
        'coherence_score': {
            'openai': 'openai/o1-preview',
            'anthropic': 'anthropic/claude-3.5-sonnet',
            'google': 'google/gemini-pro-1.5',
            'meta': 'meta-llama/llama-3.1-405b-instruct',
            'mistral': 'mistralai/mistral-large-latest',
            'x': 'x-ai/grok-beta'
        }
    }
    return best_models


def _remove_answer_count_outliers(df: pd.DataFrame) -> pd.DataFrame:
    answer_counts = df.groupby('question')['answer_num'].max()
    Q1, Q3 = answer_counts.quantile(0.25), answer_counts.quantile(0.75)
    IQR = Q3 - Q1
    normal_questions = answer_counts[
        (answer_counts >= Q1 - 1.5 * IQR) &
        (answer_counts <= Q3 + 1.5 * IQR)
    ].index
    return df[df['question'].isin(normal_questions)]


def _smooth_series(series: pd.Series, window: int = 5) -> pd.Series:
    return series.rolling(window=window, center=True, min_periods=1).mean()


def _plot_company_models(df: pd.DataFrame,
                         company: str,
                         metric: str,
                         title: str,
                         output_path: str):
    plt.figure(figsize=(16, 10))
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'serif'

    company_df = df[df['company'] == company]
    color = COMPANY_COLORS[company]

    # Add reference line based on metric
    if metric == 'coherence_score':
        plt.axhline(y=15, color='gray', linestyle=':', alpha=0.5)
    elif metric == 'embedding_score':
        plt.axhline(y=0.15, color='gray', linestyle=':', alpha=0.5)

    # Define line styles for differentiation
    line_styles = [
        {'linestyle': 'solid', 'linewidth': 1.5},
        {'linestyle': 'dashed', 'linewidth': 1.5},
        {'linestyle': 'dotted', 'linewidth': 2.0},
        {'linestyle': 'dashdot', 'linewidth': 1.5},
        # Custom dash pattern
        {'linestyle': (0, (5, 2, 1, 2)), 'linewidth': 1.5},
        {'linestyle': (0, (1, 1)), 'linewidth': 1.5},        # Dense dotted
        {'linestyle': (0, (10, 2, 1, 2)), 'linewidth': 1.5},  # Custom dash-dot
        {'linestyle': (0, (5, 1)), 'linewidth': 1.5},        # Dense dashed
    ]

    # Calculate mean values for each model and answer number
    mean_df = company_df.groupby(['model', 'answer_num'])[
        metric].mean().reset_index()

    # Plot each model's trend line with different line styles
    for idx, model in enumerate(sorted(company_df['model'].unique())):
        model_df = mean_df[mean_df['model'] == model].sort_values('answer_num')
        smoothed_values = _smooth_series(model_df[metric])

        # Get line style (cycle through styles if more models than styles)
        style = line_styles[idx % len(line_styles)]

        plt.plot(model_df['answer_num'],
                 smoothed_values,
                 color=color,
                 alpha=0.9,
                 label=model.split('/')[-1],
                 **style)  # Unpack the style dictionary

        # Add model name at end of line
        last_x = model_df['answer_num'].iloc[-1]
        last_y = smoothed_values.iloc[-1]
        plt.annotate(model.split('/')[-1],
                     xy=(last_x, last_y),
                     xytext=(5, 0),
                     textcoords='offset points',
                     va='center',
                     color=color,
                     fontsize=9,
                     fontweight='bold')

    plt.title(f"{title}\n{company.title()}", fontsize=16, pad=20)
    plt.xlabel('Answer Number', fontsize=14)
    plt.ylabel(metric.replace('_', ' ').title(), fontsize=14)

    # Add legend for line styles
    plt.legend(fontsize=9,
               title=f"{company.title()} Models",
               bbox_to_anchor=(1.05, 1),
               loc='upper left')

    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def _plot_best_models(df: pd.DataFrame,
                      metric: str,
                      best_models: dict,
                      title: str,
                      output_path: str):
    plt.figure(figsize=(16, 10))
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'serif'

    # Add reference line based on metric
    if metric == 'coherence_score':
        plt.axhline(y=15, color='gray', linestyle=':', alpha=0.5)
    elif metric == 'embedding_score':
        plt.axhline(y=0.15, color='gray', linestyle=':', alpha=0.5)

    # Get models for this metric
    metric_best_models = best_models[metric]

    # Calculate mean values for best models
    mean_df = df[df['model'].isin(metric_best_models.values())].groupby(
        ['model', 'answer_num'])[metric].mean().reset_index()

    # Plot each company's best model
    for company, model in metric_best_models.items():
        color = COMPANY_COLORS[company]
        model_df = mean_df[mean_df['model'] == model].sort_values('answer_num')
        smoothed_values = _smooth_series(model_df[metric])

        plt.plot(model_df['answer_num'],
                 smoothed_values,
                 linewidth=1.5,
                 color=color)

        # Add model name at end of line
        last_x = model_df['answer_num'].iloc[-1]
        last_y = smoothed_values.iloc[-1]
        plt.annotate(model.split('/')[-1],
                     xy=(last_x, last_y),
                     xytext=(5, 0),
                     textcoords='offset points',
                     va='center',
                     color=color,
                     fontsize=9,
                     fontweight='bold')

    plt.title(f"{title}\nBest Model per Company", fontsize=16, pad=20)
    plt.xlabel('Answer Number', fontsize=14)
    plt.ylabel(metric.replace('_', ' ').title(), fontsize=14)

    # Add company legend
    handles = [plt.Line2D([0], [0], color=color, label=company.title(), linewidth=1.5)
               for company, color in COMPANY_COLORS.items()]
    plt.legend(handles=handles,
               title='Companies',
               bbox_to_anchor=(1.05, 1),
               loc='upper left',
               fontsize=10)

    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def _compute_score_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute detailed statistics for scores including CIs and clustered SEs"""
    # First reset the index to avoid multi-index issues
    stats_df = df.groupby(['model', 'question']).agg({
        'embedding_score': ['mean', 'std', 'count'],
        'coherence_score': ['mean', 'std', 'count']
    }).reset_index()
    
    # Flatten column names
    stats_df.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col 
                       for col in stats_df.columns]
    
    # Compute confidence intervals
    for metric in ['embedding_score', 'coherence_score']:
        stats_df[f'{metric}_ci'] = stats_df.apply(
            lambda row: compute_confidence_interval(
                df[
                    (df['model'] == row['model']) & 
                    (df['question'] == row['question'])
                ][metric].values
            ),
            axis=1
        )
        
        # Add clustered standard errors
        stats_df[f'{metric}_clustered_se'] = stats_df.apply(
            lambda row: compute_clustered_se(
                df[df['model'] == row['model']], 
                metric,
                'question'
            ),
            axis=1
        )
    
    return stats_df


def analyze_questions(df: pd.DataFrame) -> dict:
    """Analyzes questions across multiple dimensions and returns insights"""
    question_stats = {}
    
    # Calculate mean scores per question
    mean_scores = df.groupby('question').agg({
        'embedding_score': ['mean', 'std'],
        'coherence_score': ['mean', 'std']
    }).round(3)
    
    # Flatten column names
    mean_scores.columns = ['_'.join(col).strip('_') for col in mean_scores.columns]
    
    # Find questions with highest/lowest scores
    question_stats['top_coherence'] = mean_scores.nlargest(5, 'coherence_score_mean')['coherence_score_mean']
    question_stats['bottom_coherence'] = mean_scores.nsmallest(5, 'coherence_score_mean')['coherence_score_mean']
    question_stats['top_embedding'] = mean_scores.nlargest(5, 'embedding_score_mean')['embedding_score_mean']
    question_stats['bottom_embedding'] = mean_scores.nsmallest(5, 'embedding_score_mean')['embedding_score_mean']
    
    # Analyze score variability
    question_stats['most_variable'] = mean_scores.nlargest(5, 'coherence_score_std')['coherence_score_std']
    question_stats['most_consistent'] = mean_scores.nsmallest(5, 'coherence_score_std')['coherence_score_std']
    
    # Add statistical analysis
    stats_df = _compute_score_statistics(df)
    question_stats['statistics'] = stats_df
    
    # Compute paired differences between top models
    top_models = df['model'].value_counts().nlargest(5).index
    paired_comparisons = []
    
    for i, model_a in enumerate(top_models):
        for model_b in top_models[i+1:]:
            comparison = analyze_paired_differences(df, model_a, model_b)
            paired_comparisons.append({
                'model_a': model_a,
                'model_b': model_b,
                **comparison
            })
    
    question_stats['paired_comparisons'] = pd.DataFrame(paired_comparisons)
    
    return question_stats


def _plot_question_analysis(df: pd.DataFrame, output_dir: Path):
    """Generates detailed question analysis visualizations"""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'serif'
    
    # Create violin plots for both metrics
    for metric in ['coherence_score', 'embedding_score']:
        plt.figure(figsize=(20, 12))
        
        # Create violin plot
        sns.violinplot(data=df, 
                      x='question', 
                      y=metric,
                      cut=0,  # Don't extend beyond data bounds
                      scale='width',  # Make all violins same width
                      inner='box')  # Show box plot inside violin
        
        # Customize appearance
        plt.xticks(rotation=45, ha='right')
        plt.xlabel('Questions', fontsize=12)
        plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
        
        # Truncate question labels if too long
        ax = plt.gca()
        labels = [label.get_text()[:50] + '...' if len(label.get_text()) > 50 
                 else label.get_text() 
                 for label in ax.get_xticklabels()]
        ax.set_xticklabels(labels)
        
        # Add summary statistics as text
        question_stats = df.groupby('question')[metric].agg(['mean', 'std', 'min', 'max'])
        for idx, (question, stats) in enumerate(question_stats.iterrows()):
            stats_text = f'μ={stats["mean"]:.2f}\nσ={stats["std"]:.2f}'
            plt.text(idx, plt.ylim()[0], stats_text,
                    ha='center', va='top', fontsize=8, rotation=45)
        
        plt.title(f'Distribution of {metric.replace("_", " ").title()} by Question',
                 pad=20, fontsize=14)
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        
        # Save high-resolution plot
        plt.savefig(output_dir / f'question_{metric}_violin.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()

    # Add correlation heatmap between questions
    plt.figure(figsize=(15, 15))
    question_correlations = df.pivot_table(
        values='coherence_score',
        index='question',
        columns='company',
        aggfunc='mean'
    ).corr()
    
    sns.heatmap(question_correlations, 
                annot=True, 
                cmap='RdYlBu_r',
                center=0,
                fmt='.2f',
                square=True)
    plt.title('Question Performance Correlation Across Companies', pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / 'question_correlations.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_question_analysis_report(stats: dict, output_dir: Path):
    """Generates a markdown report with question analysis insights"""
    report = [
        "# Question Analysis Report\n",
        "## Top Performing Questions (Coherence)",
        stats['top_coherence'].to_markdown(),
        "\n## Lowest Performing Questions (Coherence)",
        stats['bottom_coherence'].to_markdown(),
        "\n## Questions with Most Variable Responses",
        stats['most_variable'].to_markdown(),
        "\n## Questions with Most Consistent Responses",
        stats['most_consistent'].to_markdown(),
    ]
    
    with open(output_dir / 'question_analysis.md', 'w') as f:
        f.write('\n\n'.join(report))


def generate_score_plots():
    output_dir = Path('plots')
    output_dir.mkdir(exist_ok=True)

    results = _load_results()
    df = _extract_scores(results)
    df_filtered = _remove_answer_count_outliers(df)

    # Add question analysis
    question_stats = analyze_questions(df_filtered)
    _plot_question_analysis(df_filtered, output_dir)
    generate_question_analysis_report(question_stats, output_dir)

    # Get best models for each company
    best_models = _get_best_models(df_filtered)

    # Plot company-specific graphs
    for company in COMPANY_COLORS.keys():
        _plot_company_models(
            df_filtered,
            company,
            'embedding_score',
            'Embedding Dissimilarity Score Decay Over Answers',
            output_dir / f'{company}_embedding_decay.png'
        )

        _plot_company_models(
            df_filtered,
            company,
            'coherence_score',
            'Coherence Score Variation Over Answers',
            output_dir / f'{company}_coherence.png'
        )

    # Plot best models comparison
    _plot_best_models(
        df_filtered,
        'embedding_score',
        best_models,
        'Embedding Dissimilarity Score Decay Over Answers',
        output_dir / 'best_models_embedding_decay.png'
    )

    _plot_best_models(
        df_filtered,
        'coherence_score',
        best_models,
        'Coherence Score Variation Over Answers',
        output_dir / 'best_models_coherence.png'
    )


def compute_confidence_interval(data: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
    """Compute confidence interval using t-distribution"""
    n = len(data)
    mean = np.mean(data)
    se = stats.sem(data)  # Standard error of mean
    ci = t.interval(confidence, n-1, loc=mean, scale=se)
    return ci


def compute_clustered_se(data: pd.DataFrame, 
                        score_col: str,
                        cluster_col: str) -> float:
    """Compute clustered standard errors"""
    # Group by cluster and calculate means
    cluster_means = data.groupby(cluster_col)[score_col].mean()
    
    # Calculate overall mean
    overall_mean = data[score_col].mean()
    
    # Calculate clustered variance
    n = len(data)
    n_clusters = len(cluster_means)
    
    # Sum of squared deviations within clusters
    within_cluster_dev = sum(
        sum((val - cluster_means[cluster])**2 
            for val in data[data[cluster_col] == cluster][score_col])
        for cluster in cluster_means.index
    )
    
    # Between cluster variance
    between_cluster_dev = sum(
        len(data[data[cluster_col] == cluster]) * (mean - overall_mean)**2
        for cluster, mean in cluster_means.items()
    )
    
    clustered_se = np.sqrt((within_cluster_dev + between_cluster_dev) / (n * (n_clusters - 1)))
    return clustered_se


def analyze_paired_differences(df: pd.DataFrame, 
                             model_a: str, 
                             model_b: str) -> dict:
    """Analyze paired differences between two models"""
    # Get paired scores
    paired_data = df[df['model'].isin([model_a, model_b])].pivot(
        columns='model', 
        values='coherence_score'  # Using coherence_score for comparison
    )
    
    differences = paired_data[model_a] - paired_data[model_b]
    
    # Calculate statistics
    mean_diff = differences.mean()
    se = stats.sem(differences)
    ci = t.interval(0.95, len(differences)-1, loc=mean_diff, scale=se)
    t_stat, p_value = stats.ttest_rel(paired_data[model_a], paired_data[model_b])
    
    return {
        'mean_difference': mean_diff,
        'standard_error': se,
        'confidence_interval': ci,
        't_statistic': t_stat,
        'p_value': p_value
    }


if __name__ == "__main__":
    generate_score_plots()
