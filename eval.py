import json
from rag import (
    load_text,
    chunk_by_lines,
    build_index,
    retrieve,
    chunk_by_chars
)
def load_eval_set(path):
    with open(path, 'r', encoding='utf-8') as f:
        eval_set = json.load(f)
    return eval_set
def hit_at_k(index, eval_set, k):
    results = []

    for item in eval_set:
        question = item['question']
        expected = item['expected'].lower()

        chunks = retrieve(question, index, k)

        hit = any(
            expected in chunk['content'].lower()
            for chunk in chunks
        )

        results.append({
            'question': question,
            'hit': int(hit)
        })

    return results
def get_statistics(results):
    total_questions = len(results)

    if total_questions == 0:
        return {
            'total_questions': 0,
            'total_hits': 0,
            'total_misses': 0,
            'hit_rate': 0.0
        }

    total_hits = sum(item['hit'] for item in results)
    total_misses = total_questions - total_hits
    hit_rate = total_hits / total_questions

    return {
        'total_questions': total_questions,
        'total_hits': total_hits,
        'total_misses': total_misses,
        'hit_rate': hit_rate
    }
if __name__ == "__main__":

    text = load_text("info.txt")
    eval_set = load_eval_set("eval_set.json")


    configurations = {
        "line": chunk_by_lines(text),
        "char_200": chunk_by_chars(text, 200, 40),
        "char_1000": chunk_by_chars(text, 1000, 150)
    }


    k_values = [1, 3, 5]


    for config_name, chunks in configurations.items():
        print(f"\n===== {config_name} =====")

        index = build_index(chunks)

        for k in k_values:
            results = hit_at_k(index, eval_set, k)
            statistics = get_statistics(results)

            print(f"\nK = {k}")
            print(f"Hits: {statistics['total_hits']}")
            print(f"Misses: {statistics['total_misses']}")
            print(f"Hit@{k}: {statistics['hit_rate']:.2%}")