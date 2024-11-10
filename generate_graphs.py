import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from benchmark import _load_results


def plot_llm_scores_by_temperature_heatmap(results_file='results.pkl'):
    results = _load_results(results_file)
    models = results['models']
    data = []

    for model_name in models:
        for temperature in models[model_name]:
            total_novelty = 0
            count = 0
            for question in models[model_name][temperature]:
                answers = models[model_name][temperature][question]
                for answer_data in answers:
                    total_novelty += answer_data['dissimilarity_score']
                    count += 1
            avg_novelty = total_novelty / count if count > 0 else 0
            data.append({
                'Model': model_name,
                'Temperature': temperature,
                'Average Novelty': avg_novelty
            })

    df = pd.DataFrame(data)
    pivot_table = df.pivot('Model', 'Temperature', 'Average Novelty')
    temperatures = sorted(df['Temperature'].unique())
    pivot_table = pivot_table[temperatures]  # Order the columns

    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Average Novelty Scores by Model and Temperature')
    plt.xlabel('Temperature')
    plt.ylabel('Model')
    plt.tight_layout()
    plt.show()


def plot_average_novelty_bar_chart(results_file='results.pkl'):
    results = _load_results(results_file)
    models = results['models']
    data = []

    for model_name in models:
        total_novelty = 0
        count = 0
        for temperature in models[model_name]:
            for question in models[model_name][temperature]:
                answers = models[model_name][temperature][question]
                for answer_data in answers:
                    total_novelty += answer_data['dissimilarity_score']
                    count += 1
        avg_novelty = total_novelty / count if count > 0 else 0
        data.append({'Model': model_name, 'Average Novelty': avg_novelty})

    df = pd.DataFrame(data)
    df = df.sort_values('Average Novelty', ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(data=df, x='Average Novelty', y='Model', palette='viridis')
    plt.title('Average Novelty Scores by Model')
    plt.xlabel('Average Novelty Score')
    plt.ylabel('Model')
    plt.tight_layout()
    plt.show()


def plot_novelty_vs_coherence(results_file='results.pkl'):
    results = _load_results(results_file)
    models = results['models']
    data = []

    for model_name in models:
        for temperature in models[model_name]:
            for question in models[model_name][temperature]:
                answers = models[model_name][temperature][question]
                for answer_data in answers:
                    data.append({
                        'Model': model_name,
                        'Novelty Score': answer_data['dissimilarity_score'],
                        'Coherence Score': answer_data['coherence_score']
                    })

    df = pd.DataFrame(data)

    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df, x='Novelty Score',
                    y='Coherence Score', hue='Model', alpha=0.7)
    plt.title('Novelty vs. Coherence Scores')
    plt.xlabel('Novelty Score')
    plt.ylabel('Coherence Score')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def plot_coherence_score_distribution(results_file='results.pkl'):
    results = _load_results(results_file)
    models = results['models']
    data = []

    for model_name in models:
        for temperature in models[model_name]:
            for question in models[model_name][temperature]:
                answers = models[model_name][temperature][question]
                for answer_data in answers:
                    data.append({
                        'Model': model_name,
                        'Coherence Score': answer_data['coherence_score']
                    })

    df = pd.DataFrame(data)

    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df, x='Coherence Score',
                y='Model', orient='h', palette='Set2')
    plt.title('Distribution of Coherence Scores per Model')
    plt.xlabel('Coherence Score')
    plt.ylabel('Model')
    plt.tight_layout()
    plt.show()


def plot_average_processing_time(results_file='results.pkl'):
    results = _load_results(results_file)
    models = results['models']
    data = []

    for model_name in models:
        total_time = 0
        count = 0
        for temperature in models[model_name]:
            for question in models[model_name][temperature]:
                answers = models[model_name][temperature][question]
                for answer_data in answers:
                    total_time += answer_data['processing_time']
                    count += 1
        avg_time = total_time / count if count > 0 else 0
        data.append({'Model': model_name, 'Average Processing Time': avg_time})

    df = pd.DataFrame(data)
    df = df.sort_values('Average Processing Time', ascending=True)

    plt.figure(figsize=(12, 8))
    sns.barplot(data=df, x='Average Processing Time',
                y='Model', palette='pastel')
    plt.title('Average Processing Time per Model')
    plt.xlabel('Average Processing Time (seconds)')
    plt.ylabel('Model')
    plt.tight_layout()
    plt.show()


def plot_coherence_by_temperature_heatmap(results_file='results.pkl'):
    results = _load_results(results_file)
    models = results['models']
    data = []

    for model_name in models:
        for temperature in models[model_name]:
            total_coherence = 0
            count = 0
            for question in models[model_name][temperature]:
                answers = models[model_name][temperature][question]
                for answer_data in answers:
                    total_coherence += answer_data['coherence_score']
                    count += 1
            avg_coherence = total_coherence / count if count > 0 else 0
            data.append({
                'Model': model_name,
                'Temperature': temperature,
                'Average Coherence': avg_coherence
            })

    df = pd.DataFrame(data)
    pivot_table = df.pivot('Model', 'Temperature', 'Average Coherence')
    temperatures = sorted(df['Temperature'].unique())
    pivot_table = pivot_table[temperatures]

    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap='viridis')
    plt.title('Average Coherence Scores by Model and Temperature')
    plt.xlabel('Temperature')
    plt.ylabel('Model')
    plt.tight_layout()
    plt.show()


def plot_novelty_coherence_per_model(results_file='results.pkl'):
    results = _load_results(results_file)
    models = results['models']
    data = {'Model': [], 'Average Novelty': [], 'Average Coherence': []}

    for model_name in models:
        total_novelty = 0
        total_coherence = 0
        count = 0
        for temperature in models[model_name]:
            for question in models[model_name][temperature]:
                answers = models[model_name][temperature][question]
                for answer_data in answers:
                    total_novelty += answer_data['dissimilarity_score']
                    total_coherence += answer_data['coherence_score']
                    count += 1
        avg_novelty = total_novelty / count if count > 0 else 0
        avg_coherence = total_coherence / count if count > 0 else 0
        data['Model'].append(model_name)
        data['Average Novelty'].append(avg_novelty)
        data['Average Coherence'].append(avg_coherence)

    df = pd.DataFrame(data)
    df = df.set_index('Model')

    df.plot(kind='barh', stacked=True, figsize=(12, 8), colormap='Accent')
    plt.title('Average Novelty and Coherence Scores per Model')
    plt.xlabel('Score')
    plt.ylabel('Model')
    plt.tight_layout()
    plt.show()


plot_novelty_coherence_per_model()