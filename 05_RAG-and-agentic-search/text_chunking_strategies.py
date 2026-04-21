import re

# SIZE BASDED CHUNKING
# Chunk by a set number of charactesr
def chunk_by_char(text, chunk_size=150, chunk_overlap=20):
    chunks = []
    start_idx = 0

    while start_idx < len(text):
        end_idx = min(start_idx + chunk_size, len(text))

        chunk_text = text[start_idx:end_idx]
        chunks.append(chunk_text)

        start_idx = (
            end_idx - chunk_overlap if end_idx < len(text) else len(text)
        )

    return chunks

# Chunk by sentence
# Structure based chunking - chunk by sentence with a set number of sentences per chunk and a set number of sentences to overlap between chunks
def chunk_by_sentence(text, max_sentences_per_chunk=5, overlap_sentences=1):
    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks = []
    start_idx = 0

    while start_idx < len(sentences):
        end_idx = min(start_idx + max_sentences_per_chunk, len(sentences))

        current_chunk = sentences[start_idx:end_idx]
        chunks.append(" ".join(current_chunk))

        start_idx += max_sentences_per_chunk - overlap_sentences

        if start_idx < 0:
            start_idx = 0

    return chunks


# Chunk by section
def chunk_by_section(document_text):
    pattern = r"\n## "
    return re.split(pattern, document_text)

import os
_DIR = os.path.dirname(os.path.abspath(__file__))

def main ():
    with open(os.path.join(_DIR, "report.md"), "r") as f:
        text = f.read()

    # SIZE BASDED CHUNKING
    #chunks = chunk_by_char(text)


    chunks = chunk_by_sentence(text)
    #chunks = check_by

    [print(chunk + "\n----\n") for chunk in chunks]


# chunking strategies to try:
# - chunk by a set number of characters (e.g. 150 chars with 20 chars of overlap)
# - chunk by sentence (e.g. 5 sentences per chunk with 1 sentence of overlap)
# - chunk by section (e.g. split by "## " which indicates a new section in markdown)

# uv run python -m app.rag.text_chunking_strategies
if __name__ == "__main__":
    main()





