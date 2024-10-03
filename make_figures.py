import json
import matplotlib.pyplot as plt
import numpy as np
import os

def load_data(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def organize_data_by_question(data):
    question_data = {}
    for entry in data:
        question = entry['question']
        if question not in question_data:
            question_data[question] = {'coherence_scores': [], 'novelty_scores': []}
        question_data[question]['coherence_scores'].append(entry['coherence_score'])
        question_data[question]['novelty_scores'].append(entry['novelty_score'])
    return question_data

def plot_scores(question_data, output_dir='plots'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for idx, (question, scores) in enumerate(question_data.items(), 1):
        iterations = list(range(1, len(scores['coherence_scores']) + 1))
        coherence_scores = scores['coherence_scores']
        novelty_scores = scores['novelty_scores']
        
        num_iterations = len(coherence_scores)
        print(f'Question {idx} has {num_iterations} iterations.')

        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        color = 'tab:blue'
        ax1.set_xlabel('Iteration Number')
        ax1.set_ylabel('Coherence Score', color=color)
        ax1.plot(iterations, coherence_scores, color=color, marker='o', label='Coherence Score')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylim(0, 10)  # Assuming coherence scores are between 0 and 10
        
        ax2 = ax1.twinx()  # Instantiate a second axes that shares the same x-axis
        
        color = 'tab:red'
        ax2.set_ylabel('Novelty Score', color=color)
        ax2.plot(iterations, novelty_scores, color=color, marker='x', linestyle='--', label='Novelty Score')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim(0, 1)  # Novelty scores between 0 and 1
        
        plt.title(f'Question {idx}: Coherence and Novelty Scores vs Iteration\nTotal Iterations: {num_iterations}')
        fig.tight_layout()  # Adjust layout to prevent clipping
        
        # Save the plot
        plot_filename = os.path.join(output_dir, f'question_{idx}_scores.png')
        plt.savefig(plot_filename)
        plt.close()
        print(f'Saved plot for Question {idx} to {plot_filename}')


def plot_all_scores(question_data, output_dir='plots'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    max_iterations = max(len(scores['coherence_scores']) for scores in question_data.values())
    
    # Initialize lists to collect scores per iteration
    coherence_scores_per_iter = [[] for _ in range(max_iterations)]
    novelty_scores_per_iter = [[] for _ in range(max_iterations)]
    
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # Plot individual sequences
    for idx, (question, scores) in enumerate(question_data.items(), 1):
        iterations = np.arange(1, len(scores['coherence_scores']) + 1)
        coherence_scores = scores['coherence_scores']
        novelty_scores = scores['novelty_scores']
        
        # Update scores per iteration
        for i in range(len(coherence_scores)):
            coherence_scores_per_iter[i].append(coherence_scores[i])
            novelty_scores_per_iter[i].append(novelty_scores[i])
        
        # Plot individual coherence scores
        ax1.plot(iterations, coherence_scores, color='tab:blue', alpha=0.3)
        
    ax1.set_xlabel('Iteration Number')
    ax1.set_ylabel('Coherence Score', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_ylim(0, 10)  # Adjust based on your coherence score range
    
    # Plot average coherence scores
    avg_coherence_scores = [np.mean(scores) if scores else np.nan for scores in coherence_scores_per_iter]
    iterations = np.arange(1, len(avg_coherence_scores) + 1)
    ax1.plot(iterations, avg_coherence_scores, color='tab:blue', marker='o', label='Average Coherence Score')
    
    # Create second y-axis for novelty scores
    ax2 = ax1.twinx()
    
    for idx, (question, scores) in enumerate(question_data.items(), 1):
        iterations = np.arange(1, len(scores['novelty_scores']) + 1)
        novelty_scores = scores['novelty_scores']
        
        # Plot individual novelty scores
        ax2.plot(iterations, novelty_scores, color='tab:red', alpha=0.3, linestyle='--')
        
    ax2.set_ylabel('Novelty Score', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.set_ylim(0, 1)  # Novelty scores range between 0 and 1
    
    # Plot average novelty scores
    avg_novelty_scores = [np.mean(scores) if scores else np.nan for scores in novelty_scores_per_iter]
    ax2.plot(iterations, avg_novelty_scores, color='tab:red', marker='x', linestyle='--', label='Average Novelty Score')
    
    # Add legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.title('Coherence and Novelty Scores vs Iteration Across All Questions')
    fig.tight_layout()
    
    # Save the plot
    plot_filename = os.path.join(output_dir, 'all_questions_scores.png')
    plt.savefig(plot_filename)
    plt.close()
    print(f'Saved combined plot to {plot_filename}')



def main():
    filename = 'benchmark_gpt-4-turbo_1727924682.json'
    data = load_data(filename)
    question_data = organize_data_by_question(data)
    plot_all_scores(question_data)

if __name__ == "__main__":
    main()
