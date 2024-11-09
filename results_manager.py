from threading import Lock
import pickle

_results = None
_results_lock = Lock()

def get_results(filename: str = 'results.pkl') -> dict:
    global _results
    with _results_lock:
        if _results is None:
            try:
                with open(filename, 'rb') as f:
                    _results = pickle.load(f)
            except FileNotFoundError:
                _results = {'models': {}}
        return _results

def save_results(filename: str = 'results.pkl'):
    with _results_lock:
        with open(filename, 'wb') as f:
            pickle.dump(_results, f) 