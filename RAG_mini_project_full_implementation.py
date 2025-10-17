"""
RAG Mini-Project Implementation
Files expected: './data/AST-1.txt' and './data/AST-2.txt' (or PDF/MD).

This single-file script demonstrates steps 1-6 from the assignment:
1) Load docs
2) Chunk documents (multiple strategies)
3) Create embeddings (choose from MTEB-recommended models)
4) Store embeddings in FAISS vectorstore
5) Retrieval + LLM (with optional MMR)
6) Gradio app
7) Deployment instructions for Hugging Face Spaces

Instructions:
- Create a virtualenv and `pip install -r requirements.txt` (see below)
- Put your AST-1 and AST-2 files in ./data
- Run this script: `python RAG_mini_project_full_implementation.py`

### Files to include in your Space repository:
- `RAG_mini_project_full_implementation.py` (this script)
- Folder `data/` with your documents (AST-1.txt, AST-2.txt)
- Requirements (save to requirements.txt):

langchain>=0.0.300
sentence-transformers>=2.2.2
huggingface-hub
faiss-cpu
gradio
tqdm
fsspec
PyPDF2
transformers>=4.33.0
openai (optional, for OpenAI LLMs)


5. In the Space’s **Settings**, set the **entry point** to this script (or rename it to `app.py`).
6. Commit and push changes.
7. The Space will automatically build and start; after deployment, you’ll see the Gradio app appear live at:
`https://huggingface.co/spaces/<your-username>/<your-space-name>`
8. You can share the link for public access or keep it private.

Notes on embedding model choice:
- Use MTEB leaderboard as a guide: popular choices include `all-MiniLM-L6-v2`, `all-mpnet-base-v2`, and larger models for higher quality.
- If you have GPU and HF token, you can use larger models like 'sentence-transformers/all-mpnet-base-v2' or community models reported on MTEB.

This script keeps things modular so you can swap components.

"""

import os
from typing import List, Tuple

# --------- Config ---------
DATA_DIR = "./data"
DOC_FILES = ["AST-1.pdf", "AST-2.pdf"]  # change to .pdf or .md if needed
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # small & fast; try all-mpnet-base-v2 for higher quality
VECTORSTORE_DIR = "./vectorstore"
USE_OPENAI = False  # set True and configure OPENAI_API_KEY env var to use OpenAI completion/Chat API
LLM_BACKEND = "huggingface"  # "openai" or "huggingface"
HF_LLM_MODEL = "google/flan-t5-small"  # for local generation; change to your preference
TOP_K = 5

# Hyperparameters to tune
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
MMR_LAMBDA = 0.7

# --------- Helpers: Loading Documents ---------

def load_txt_files(data_dir: str, filenames: List[str]) -> List[Tuple[str,str]]:
    """Return list of (source, text)"""
    docs = []
    for fn in filenames:
        path = os.path.join(data_dir, fn)
        if not os.path.exists(path):
            print(f"Warning: {path} not found. Skipping.")
            continue
        if fn.lower().endswith('.pdf'):
            from PyPDF2 import PdfReader
            reader = PdfReader(path)
            text = "\n".join([p.extract_text() or "" for p in reader.pages])
            docs.append((fn, text))
        else:
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
            docs.append((fn, text))
    return docs

# --------- Chunking Strategies ---------

def chunk_text_charwise(text: str, chunk_size: int=CHUNK_SIZE, overlap: int=CHUNK_OVERLAP) -> List[str]:
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + chunk_size, L)
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def chunk_text_sentencewise(text: str, chunk_size: int=CHUNK_SIZE, overlap: int=CHUNK_OVERLAP) -> List[str]:
    # naive sentence splitter by punctuation. For production use nltk/spacy.
    import re
    sents = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    cur = []
    cur_len = 0
    for s in sents:
        if cur_len + len(s) <= chunk_size or not cur:
            cur.append(s)
            cur_len += len(s)
        else:
            chunks.append(' '.join(cur))
            # start new
            cur = [s]
            cur_len = len(s)
    if cur:
        chunks.append(' '.join(cur))
    # apply overlap by merging last tokens from previous chunk if needed (simple)
    return chunks

# --------- Create embeddings and vectorstore ---------

def create_embeddings_and_store(docs: List[Tuple[str,str]], embedding_model_name: str=EMBEDDING_MODEL, vectorstore_dir: str=VECTORSTORE_DIR):
    from sentence_transformers import SentenceTransformer
    import numpy as np
    import faiss
    import json

    model = SentenceTransformer(embedding_model_name)

    # Prepare documents and metadata
    all_texts = []
    metadatas = []
    for src, text in docs:
        # try sentencewise chunking first, fall back to charwise
        chunks = chunk_text_sentencewise(text)
        if len(chunks) == 0:
            chunks = chunk_text_charwise(text)
        for i, c in enumerate(chunks):
            all_texts.append(c)
            metadatas.append({"source": src, "chunk": i})

    print(f"Total chunks: {len(all_texts)}")

    # compute embeddings in batches
    batch_size = 64
    embeddings = model.encode(all_texts, batch_size=batch_size, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product; use IndexFlatL2 for Euclidean
    # normalize vectors for cosine similarity
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    # save index and metadata
    os.makedirs(vectorstore_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(vectorstore_dir, 'faiss.index'))
    with open(os.path.join(vectorstore_dir, 'metadatas.json'), 'w', encoding='utf-8') as f:
        json.dump(metadatas, f, ensure_ascii=False, indent=2)
    with open(os.path.join(vectorstore_dir, 'texts.json'), 'w', encoding='utf-8') as f:
        json.dump(all_texts, f, ensure_ascii=False, indent=2)

    print('Saved vectorstore to', vectorstore_dir)

# --------- Retrieval (kNN and MMR) ---------

def load_vectorstore(vectorstore_dir: str=VECTORSTORE_DIR):
    import faiss, json
    idx = faiss.read_index(os.path.join(vectorstore_dir, 'faiss.index'))
    with open(os.path.join(vectorstore_dir, 'metadatas.json'), 'r', encoding='utf-8') as f:
        metadatas = json.load(f)
    with open(os.path.join(vectorstore_dir, 'texts.json'), 'r', encoding='utf-8') as f:
        texts = json.load(f)
    return idx, metadatas, texts


def embed_query(query: str, embedding_model_name: str=EMBEDDING_MODEL):
    from sentence_transformers import SentenceTransformer
    import numpy as np
    import faiss
    model = SentenceTransformer(embedding_model_name)
    emb = model.encode([query])
    emb = np.array(emb).astype('float32')
    faiss.normalize_L2(emb)
    return emb


def retrieve_knn(query: str, k: int=TOP_K):
    import numpy as np
    idx, metadatas, texts = load_vectorstore()
    q_emb = embed_query(query)
    D, I = idx.search(q_emb, k)
    results = []
    for score, i in zip(D[0], I[0]):
        results.append({"score": float(score), "text": texts[i], "meta": metadatas[i]})
    return results


def mmr_rerank(query: str, k: int=10, fetch_k: int=50, lambda_mult: float=MMR_LAMBDA):
    # Implement a simple MMR: fetch top fetch_k, then greedily select up to k with MMR
    import numpy as np
    from sentence_transformers import SentenceTransformer

    idx, metadatas, texts = load_vectorstore()
    model = SentenceTransformer(EMBEDDING_MODEL)

    q_emb = model.encode([query])[0]
    import faiss
    q_emb = np.array([q_emb]).astype('float32')
    faiss.normalize_L2(q_emb)

    D, I = idx.search(q_emb, fetch_k)
    candidate_embs = []
    candidate_texts = []
    candidate_meta = []
    for i in I[0]:
        candidate_texts.append(texts[i])
        candidate_meta.append(metadatas[i])
    candidate_embs = model.encode(candidate_texts)
    # normalize
    from numpy.linalg import norm
    candidate_embs = candidate_embs / (np.linalg.norm(candidate_embs, axis=1, keepdims=True)+1e-12)
    q_emb_norm = q_emb[0] / (np.linalg.norm(q_emb[0]) + 1e-12)

    selected = []
    selected_idx = []
    # start by picking the most relevant
    sims = (candidate_embs @ q_emb_norm).tolist()
    first = int(np.argmax(sims))
    selected.append(candidate_texts[first])
    selected_idx.append(first)

    while len(selected) < k and len(selected_idx) < len(candidate_texts):
        mmr_scores = []
        for i, emb in enumerate(candidate_embs):
            if i in selected_idx:
                mmr_scores.append(-9999)
                continue
            sim_to_query = float(np.dot(emb, q_emb_norm))
            sim_to_selected = max([float(np.dot(emb, candidate_embs[s])) for s in selected_idx]) if selected_idx else 0.0
            mmr_score = lambda_mult * sim_to_query - (1 - lambda_mult) * sim_to_selected
            mmr_scores.append(mmr_score)
        nxt = int(np.argmax(mmr_scores))
        selected.append(candidate_texts[nxt])
        selected_idx.append(nxt)
    results = []
    for idx_sel in selected_idx:
        results.append({"text": candidate_texts[idx_sel], "meta": candidate_meta[idx_sel]})
    return results

# --------- LLM integration (simple) ---------

def generate_answer_with_context(question: str, contexts: List[str], llm_backend: str=LLM_BACKEND):
    # Compose prompt
    prompt = "You are an assistant. Use the following context to answer the question. Do not hallucinate.\n\n"
    for i, c in enumerate(contexts):
        prompt += f"[CONTEXT {i}] {c}\n"
    prompt += "\nQUESTION: " + question + "\nANSWER:"

    if llm_backend == 'openai' and USE_OPENAI:
        import openai
        openai.api_key = os.environ.get('OPENAI_API_KEY')
        resp = openai.ChatCompletion.create(model='gpt-4o-mini', messages=[{"role":"user","content":prompt}], max_tokens=512)
        return resp['choices'][0]['message']['content']
    else:
        # use HuggingFace pipeline
        from transformers import pipeline
        pipe = pipeline('text-generation', model=HF_LLM_MODEL, device=-1, max_new_tokens=256)
        out = pipe(prompt, do_sample=False)[0]['generated_text']
        # when using generation models, remove the prompt prefix if returned
        if out.startswith(prompt):
            out = out[len(prompt):]
        return out

# --------- Example Questions (Step 5) ---------
EXAMPLE_QUESTIONS = [
    "What is a production code?",
    "Why we need data_manger.py?",
    "Explain Project folder structure.",
    "What are types of tests?",
    "Explain the steps to configure the project and execute it."
]

# --------- Gradio App ---------

def build_gradio_app(): 
    import gradio as gr
    import pandas as pd

    def answer_query(query: str, use_mmr: bool, k: int):
        if use_mmr:
            retrieved = mmr_rerank(query, k=k, fetch_k=50)
            contexts = [r['text'] for r in retrieved]
        else:
            retrieved = retrieve_knn(query, k=k)
            contexts = [r['text'] for r in retrieved]
        answer = generate_answer_with_context(query, contexts)
        return answer, retrieved

    with gr.Blocks() as demo:
        gr.Markdown("# RAG Demo — AST-1 & AST-2 knowledge base")
        with gr.Row():
            inp = gr.Textbox(lines=2, placeholder='Enter your question here...')
            mmr = gr.Checkbox(label='Use MMR reranking', value=False)
            kslider = gr.Slider(minimum=1, maximum=10, step=1, value=5, label='Top-k')
        btn = gr.Button('Get Answer')
        out = gr.Markdown()
        with gr.Accordion('Retrieved chunks (debug)', open=False):
            retrieved_output = gr.Dataframe(headers=['score', 'text', 'meta'], datatype=['number', 'str', 'json'])

        # ---- Updated callback ----
        def on_click(q, use_mmr, k):
            ans, retrieved = answer_query(q, use_mmr, int(k))
            # format retrieved for dataframe
            rows = []
            for r in retrieved:
                row = [
                    r.get('score', None),
                    r['text'][:300] + ('...' if len(r['text']) > 300 else ''),
                    r.get('meta', {})
                ]
                rows.append(row)
            df = pd.DataFrame(rows, columns=['score', 'text', 'meta'])
            # return values in the same order as outputs
            return '### Answer\n' + ans, df

        # Pass outputs to the function, not empty list
        btn.click(on_click, inputs=[inp, mmr, kslider], outputs=[out, retrieved_output])

    return demo


# --------- Main flow to run all steps ---------
if __name__ == '__main__':
    docs = load_txt_files(DATA_DIR, DOC_FILES)
    if not docs:
        print('No documents loaded. Please add AST-1.txt and AST-2.txt in ./data and re-run.')
    else:
        # Step 2+3: create embeddings and store vectorstore
        create_embeddings_and_store(docs)

        # Step 5: show example Q&A
        print('\n=== Example Q&A from RAG system ===')
        for q in EXAMPLE_QUESTIONS:
            print('\nQ:', q)
            retrieved = retrieve_knn(q, k=TOP_K)
            contexts = [r['text'] for r in retrieved]
            ans = generate_answer_with_context(q, contexts)
            print('A:', ans[:500], '...')

        # Step 6: launch Gradio app
        app = build_gradio_app()
        app.launch(server_name='0.0.0.0', share=False)
