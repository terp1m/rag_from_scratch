import ollama

EMBEDDING_MODEL = 'embeddinggemma-2'
LANGUAGE_MODEL = 'qwen3:8b'


def load_text(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def chunk_by_lines(text):
    lines = text.splitlines()
    chunks = []
    chunk_id = 0
    for item in lines:
        cleaned = item.strip()
        if not cleaned:
            continue
        chunks.append({
            'id': chunk_id,
            'content': cleaned
        })
        chunk_id += 1
    return chunks


def chunk_by_chars(text, chunk_size=1000, overlap_size=150):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap_size < 0:
        raise ValueError("overlap_size must be >= 0")
    if overlap_size >= chunk_size:
        raise ValueError("overlap_size must be less than chunk_size")

    difference = chunk_size - overlap_size
    start = 0
    chunk_id = 0
    chunks = []
    while start < len(text):
        chunks.append({
            'id': chunk_id,
            'content': text[start:start + chunk_size]
        })
        if start + chunk_size >= len(text):
            break
        start += difference
        chunk_id += 1
    return chunks


def build_index(chunks):
    if not chunks:
        return []

    texts = [chunk['content'] for chunk in chunks]
    embeddings = ollama.embed(model=EMBEDDING_MODEL, input=texts)['embeddings']

    if len(chunks) != len(embeddings):
        raise ValueError("Mismatch between chunks count and embeddings count")

    indexed_chunks = []
    for chunk, embedding in zip(chunks, embeddings):
        new_chunk = chunk.copy()
        new_chunk['embedding'] = embedding
        indexed_chunks.append(new_chunk)

    return indexed_chunks


def cosine_similarity(a, b):
    if len(a) != len(b):
        raise ValueError("Vectors must have the same length")

    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x ** 2 for x in a) ** 0.5
    norm_b = sum(x ** 2 for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def retrieve(query, index, k=3):
    if k <= 0:
        raise ValueError("k must be > 0")
    if not index:
        return []

    query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]

    results = []
    for item in index:
        similarity = cosine_similarity(query_embedding, item['embedding'])

        result_item = {key: val for key, val in item.items() if key != 'embedding'}
        result_item['similarity'] = similarity
        results.append(result_item)

    results.sort(key=lambda x: x['similarity'], reverse=True)
    return results[:k]
if __name__ == "__main__":
    text = load_text("info.txt")
    chunks = chunk_by_chars(text)
    index = build_index(chunks)

    results = retrieve("your question", index)


