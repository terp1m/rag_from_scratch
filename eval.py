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
def analyze_miss(index, question, expected):
    expected = normalize(expected)


    ranked_chunks = retrieve(question, index, len(index))


    correct_chunk = None
    correct_rank = None

    for rank, chunk in enumerate(ranked_chunks, start=1):
        if expected in normalize(chunk['content']):
            correct_chunk = chunk
            correct_rank = rank
            break

    # Получаем top-1
    top1 = ranked_chunks[0] if ranked_chunks else None

    return {
        'question': question,
        'top1_content': top1['content'] if top1 else None,
        'top1_similarity': top1['similarity'] if top1 else None,
        'correct_rank': correct_rank,
        'correct_similarity': correct_chunk['similarity'] if correct_chunk else None,
        'correct_content': correct_chunk['content'] if correct_chunk else None
    }
def normalize(s):
    s = (" ".join(s.split()).lower())
    return s
def coverage(chunks, eval_set):
    if not eval_set:
        return 0.0, []
    results = []
    uncovered = []
    for item in eval_set:
        expected = normalize(item['expected'])
        covered = any(
            expected in normalize(chunk['content'])
            for chunk in chunks
        )
        if not covered:
            uncovered.append(item)
        results.append(int(covered))
    return sum(results) / len(results), uncovered
def hit_at_k(index, eval_set, k):
    results = []

    for item in eval_set:
        question = item['question']
        expected = normalize(item['expected'])

        chunks = retrieve(question, index, k)

        hit = any(
            expected in normalize(chunk['content'])
            for chunk in chunks
        )
        chars = sum(len(chunk['content']) for chunk in chunks)
        results.append({
            'question': question,
            'expected': expected,
            'hit': int(hit),
            'chars': chars
        })

    return results
def get_statistics(results):
    total_questions = len(results)

    if total_questions == 0:
        return {
            'total_questions': 0,
            'total_hits': 0,
            'total_misses': 0,
            'hit_rate': 0.0,
            'avg_chars': 0
        }

    total_hits = sum(item['hit'] for item in results)
    total_misses = total_questions - total_hits
    hit_rate = total_hits / total_questions
    avg_chars = sum(item['chars'] for item in results) / total_questions

    return {
        'total_questions': total_questions,
        'total_hits': total_hits,
        'total_misses': total_misses,
        'hit_rate': hit_rate,
        'avg_chars': avg_chars
    }
if __name__ == "__main__":

    text = load_text("info.txt")
    eval_set = load_eval_set("eval_set.json")


    configurations = {
        "line": chunk_by_lines(text),
        "char_200": chunk_by_chars(text, 200, 40),
        "char_1000": chunk_by_chars(text, 1000, 150)
    }
    for name, chunks in configurations.items():
        print(coverage(chunks,eval_set))

    k_values = [1, 3, 5]


    for config_name, chunks in configurations.items():
        print(f"\n===== {config_name} =====")

        index = build_index(chunks)

        for k in k_values:
            results = hit_at_k(index, eval_set, k)
            misses = [
                item for item in results
                if not item['hit']
            ]
            statistics = get_statistics(results)
            for miss in misses:
                analysis = analyze_miss(
                    index,
                    miss['question'],
                    miss['expected']
                )

                print("\n--- MISS ANALYSIS ---")
                print(f"Question: {analysis['question']}")
                print(f"Top-1 similarity: {analysis['top1_similarity']}")
                print(f"Correct rank: {analysis['correct_rank']}")
                print(f"Correct similarity: {analysis['correct_similarity']}")
                print(f"Correct chunk: {analysis['correct_content']}")


            print(f"\nK = {k}")
            print(f"Hits: {statistics['total_hits']}")
            print(f"Misses: {statistics['total_misses']}")
            print(f"Hit@{k}: {statistics['hit_rate']:.2%}")
            print(f"Avg chars: {statistics['avg_chars']:.0f}")