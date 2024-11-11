from benchmark import _load_results

def print_model_results():
    results = _load_results()
    models = results.get('models', {})

    for model_name, temperatures in models.items():
        print(f"Model: {model_name}")
        for temperature, questions in temperatures.items():
            print(f"  Temperature: {temperature}")
            for question, answers in questions.items():
                print(f"    Question: {question}")
                for answer_data in answers:
                    print(_format_answer_data(answer_data))
                print()

def _format_answer_data(answer_data):
    answer_num = answer_data.get('answer_num', 'N/A')
    answer = answer_data.get('answer', '')
    embedding_dissimilarity_score = answer_data.get('embedding_dissimilarity_score', 'N/A')
    coherence_score = answer_data.get('coherence_score', 'N/A')
    llm_dissimilarity_score = answer_data.get('llm_dissimilarity_score', None)
    processing_time = answer_data.get('processing_time', 'N/A')

    formatted_data = []
    formatted_data.append(f"      Answer #{answer_num}:")
    formatted_data.append(f"        Answer: {answer}")
    formatted_data.append(f"        Embedding Dissimilarity Score: {embedding_dissimilarity_score}")
    formatted_data.append(f"        Coherence Score: {coherence_score}")
    if llm_dissimilarity_score is not None:
        formatted_data.append(f"        LLM Dissimilarity Score: {llm_dissimilarity_score}")
    formatted_data.append(f"        Processing Time: {processing_time}")
    return '\n'.join(formatted_data)

if __name__ == "__main__":
    print_model_results()