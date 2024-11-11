from benchmark import _load_results

def print_questions_per_model():
    results = _load_results()
    models = results.get('models', {})

    for model_name, temperatures in models.items():
        print(f"Model: {model_name}")
        questions = set()

        for temperature_data in temperatures.values():
            questions.update(temperature_data.keys())

        print("Questions:")
        for question in questions:
            print(f"  - {question}")
        print()

if __name__ == "__main__":
    print_questions_per_model()