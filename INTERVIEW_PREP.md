# INTERVIEW PREP — RAG Evaluation & Benchmarking Pipeline

> Project: https://github.com/aanandkishor42/rag-evaluation-benchmark
> Live app: Streamlit Cloud (public). Local: `streamlit run app.py`
> Stack: Python · Streamlit · LangChain · ChromaDB · RAGAS · Ollama (local) · Groq API (cloud) · FastEmbed

---

## 0) 30-SECOND ELEVATOR PITCH (memorise this)

> "I built a web app where anyone uploads their own document, types their own
> questions, and the app answers those questions using RAG and then scores the
> answers with RAGAS — 4 accuracy metrics — live, row by row. It runs 100% free:
> locally with Ollama, and publicly on Streamlit Cloud using Groq's free API and
> free local embeddings. It is on GitHub and deployed publicly."

---

## 1) MUST-KNOW NUMBERS (memorise)
| Thing | Value | Why |
|---|---|---|
| Chunk size / overlap | 600 / 150 | 600 chars per chunk, 150 overlap so sentences don't get cut |
| top_k (retrieval) | 5 | raised 3 → 5 so the right chunk never misses |
| Groq free tier | 1,000 req/day, 30 req/min per key | judge uses it → parallel=2, retries=5 to stay under limit |
| RAGAS metrics | 4 | context_precision, context_recall, faithfulness, answer_relevancy (LLM-as-judge) |
| Embeddings | local: nomic-embed-text (Ollama); cloud: bge-small-en-v1.5 (FastEmbed, 384-dim) | free, no API call |
| Judge model | local: llama3.2:3b; cloud: gpt-oss-20b | openai-compatible |

One real sample result you can quote (bundled demo, baseline-600-80):
**context_precision 1.0 · context_recall 0.84 · faithfulness 0.97 · answer_relevancy 0.87**
("everything ~0.8-1.0, so the system works correctly")

---

# EASY QUESTIONS (they will definitely ask)

## Q1. What is this project about?
"My tool measures how accurate a chatbot's answers are, without guessing. Chatbots sometimes
make things up (hallucinate). So I built a system where you upload a document, type questions,
and it answers using RAG and scores each answer on 4 accuracy metrics. Anyone can use it free."
**Power phrase:** "It is an **LLM-as-judge accuracy measurement pipeline**."

## Q2. What is RAG? (explain simply)
"RAG = Retrieval-Augmented Generation. Analogy: imagine you have to answer an exam but first you
are allowed to open the textbook — RAG does exactly that. First (R)etrieval: find the right
paragraph from your documents. Then (G)eneration: the LLM answers using ONLY that paragraph.
So the model never answers from memory — it always checks the document first."

**Cross-question: "Why RAG instead of fine-tuning?"**
"Fine-tuning changes the model's weights — you need lots of data and retrain every time a document
changes. RAG needs no retraining: drop in a new file and answers update instantly, like a search
engine. RAG gives you current + controllable answers; fine-tune is for style or specific behaviour."

## Q3. What are embeddings and a vector store?
"An embedding turns text into a list of numbers (a vector) that captures its meaning — similar
sentences get similar numbers. A vector store is a database that searches by meaning, not keywords:
'find the chunk closest in numbers to my question.' That is how retrieval works."

## Q4. What is chunking, and why chunk size + overlap?
"Documents are cut into small pieces (600 characters) because the LLM can't see a whole PDF at once
— it receives the few most relevant pieces. Overlap 150 means the next chunk repeats the last 150
characters, so a sentence doesn't get cut in half. If half a sentence is in one chunk and half in
another, the answer gets missed — overlap fixes that."

## Q5. What are the 4 RAGAS metrics? (simple)
- **context_precision** — of the chunks that were retrieved, how many were actually relevant?
- **context_recall** — did the top chunks include the one containing the real answer? (needs a ground-truth `reference`)
- **faithfulness** — are all claims in the answer supported by the document? ← the **hallucination check, most important**
- **answer_relevancy** — does the answer actually address the question asked?
All scored 0-1, 1 is perfect.

## Q6. Why did you choose free tools?
"Coding compilers love cost-awareness. I wanted the project to be free and public. Embeddings:
Ollama/FastEmbed (local, free). Judge: Groq free tier. Hosting: Streamlit Cloud free tier.
Everything also runs locally with no internet. Cost = $0."

## Q7. Why Streamlit?
"Streamlit is the easiest way to make a data app in Python — a button, uploader or slider is one
line of code. My whole pipeline was already Python, so the UI lives in the same language, and
Streamlit Cloud hosts it free directly from GitHub."

---

# MEDIUM QUESTIONS

## Q8. Walk me through a benchmark run (practice in this order)
1. User uploads document + questions (or bundled demo data).
2. **Indexing:** document → chunks (600/150) → embeddings → Chroma vector store.
3. **Answering:** for each question, `similarity_search` → top 5 chunks → prompt (context + question) → LLM (local llama3.2 / cloud groq) → answer.
4. **Scoring:** answer + retrieved chunks + (reference if provided) → RAGAS scores with the judge LLM.
5. **Report:** rows stream live per question, aggregate scores, CSV download.
6. Saved to SQLite (`results/benchmark.db`).

## Q9. How does the LLM-as-judge actually score, e.g. faithfulness?
"For faithfulness, the judge LLM does two steps: (1) list every claim in the answer, (2) check
each claim against the retrieved document chunks. If all claims are supported → 1.0. If the model
added something not in the document → that claim fails. That is how hallucination is caught."

## Q10. context_precision vs context_recall — the classic question
- **Precision:** "Of the top-5 chunks I retrieved, how many were relevant?"
- **Recall:** "Did the chunk with the correct answer even make it into the top-5?"
Analogy: search engine — precision = are the results relevant? recall = is the right result in the list?
**Trade-off:** increasing top_k helps recall but can hurt precision (more irrelevant chunks come in).
In my repo, I raised top_k 3→5, which cut the 'I don't know' refusals (better recall).

## Q11. "I don't have enough information" even though the answer was in the PDF — the debugging story
"This was my most interesting bug. Root cause: retrieval only got the top 3 chunks, but the passage
with the answer was in the 4th chunk or split across a chunk boundary — so the model genuinely never
saw it, and honestly said 'I don't know' (my prompt forbids guessing). Fixes: top_k 3→5, overlap
80→150, and I softened the prompt to say 'combine all chunks, don't refuse just because one chunk
lacks the answer.' Now the same question answers correctly."
**Probe: "So you also had to change the prompt?"**
"Yes — that's the design principle: retrieval quality and prompt instruction together decide answer
quality. The model wasn't lying (good), it just never received the right chunk (bad retrieval). I
fixed both."

## Q12. The "judge didn't score / quota" issue
"Cloud Groq free tier allows 30 requests per minute. I originally let RAGAS run 8 judge calls in
parallel — they instantly tripped the rate limit, so all four metrics came back empty. Fixes:
parallel workers 2, retries 5, and before every run a cheap one-call 'ping' tells us whether the
judge is reachable. Also, if a question's scores all come back empty, the app auto-retries the
scoring. (Small local models also have flaky JSON output — the retry fixes that too.)"

## Q13. What do the scores tell you? (interpret real numbers)
"In one real run: context_precision 1.0 = all retrieved chunks were relevant; context_recall 0.84
= the right chunk was in the list 84% of the time; faithfulness 0.97 = answers stay true to the
document — hallucination is rare. Because every run stores its config, I can compare configurations
(chunk size × overlap × top_k) and catch regressions, like a CI test for a RAG system."

## Q14. How do you catch regressions across runs?
"Every run saves experiment name, config and scores into SQLite with a timestamp. `python
run_benchmark.py --history` lists past runs, and the Scores tab shows latest per experiment. It is
effectively a unit test for the RAG pipeline."

## Q15. Why do you need the reference (expected answer) column?
"`context_recall` (and ground-truth reasoning in precision) needs to know the correct answer ahead
of time. The reference tells RAGAS what the right answer is, so it can check whether retrieval
actually surfaced that information. The app also auto-builds `reference_contexts` — it retrieves the
chunks most similar to the reference text — so users only need to type an expected answer and
context_recall works."

## Q16. Local Ollama vs Cloud Groq trade-offs
"Local (Ollama) is free, unlimited and private — data never leaves the machine — but slow
(~1-3 min per question) and needs a computer. Cloud Groq is ~50x faster (20-60s per question) and
great for demos/interviews, but the free tier gives you 1,000 requests/day per API key — so for a
demo I keep a fresh key ready."

---

# HARD / TRICKY / WHERE YOU WILL GET STUCK

## Q17. What are the limitations of this project? (say at least 3 — honesty is a senior signal)
1. **LLM-as-judge is itself an LLM** — scoring is not a published academic benchmark; it varies by
   judge model and is slower, and a weak judge gives weak scores.
2. **Ground-truth dependency** — context_recall/precision need a good `reference`; real-world docs
   often don't have ground truth for every question (some cells stay blank by design).
3. **Fixed chunking** — 600-char chunks may cut semantic boundaries (improvement: semantic or
   recursive chunking).
4. **API limits + ephemeral storage** — free-tier rate limits, and Streamlit Cloud storage is
   per-session, so I committed `sample_results/` so the public app always shows scores.
5. **Cost/time of judging** — each question costs ~4-5 judge calls; large corpora get slow/expensive.

## Q18. How would you improve context_precision if it is low?
"Low precision means irrelevant chunks are being retrieved. Options: reduce top_k, smaller chunks,
add a reranker (fetch top 10-15, re-rank down to top 5), hybrid search (vector + BM25 keyword) to
boost exact-term matches, or split retrieval per question type."

## Q19. How do you stop hallucination?
"Three layers: (1) prompt — 'answer only from context; if absent, say you don't know',
(2) good retrieval so the context is complete, (3) faithfulness metric to actually measure and
catch it. Measurement is the first step to fixing."

## Q20. Secure / multi-user thinking?
"Uploads restricted to txt/md/pdf, files go to a session temp folder (never committed), API keys are
env/secrets only, never in the repo, and it runs in Streamlit's sandbox. Improvements next:
file-size limits, per-user isolation." *(Stay humble: "demo-grade but the patterns are right.")*

## Q21. How do you handle no-reference questions?
"`context_recall` needs a reference, so for questions without one that column is skipped and shown
blank (`-`). The app's preview shows a `has reference` ✓/— column so the user knows which questions
will get a recall score. Faithfulness, answer_relevancy and context_precision still work without a
reference. Design choice: a blank is better than a wrong score."

## Q22. How did you test your code?
"Every fix: `py_compile` on changed files, `run_benchmark.py --smoke` (offline, fast, needs no API
key), a local HTTP health-check of the Streamlit app, and a real one-question RAGAS run locally to
confirm scores (e.g., context_recall came out 1.0). Smoke test is the cheap first signal, full run
is the real verification."

## Q23. Why not judge by a single average score?
"An average hides the precision/recall tension — one config can raise precision and lower recall.
So I always present metrics side by side and choose a config based on the use case (e.g., high
faithfulness for something legal, high recall for support Q&A)."

---

# CROSS-QUESTION BANK (what they ask AFTER your answer)

| If you say… | They will ask… |
|---|---|
| "I used RAG" | "When would you prefer fine-tuning?" |
| "I set top_k=5" | "What would you test to optimise it?" (grid: 3/5/8) |
| "faithfulness came out high" | "How is faithfulness actually measured?" |
| "Groq free tier" | "What are the limits and how did you handle failures?" (30 RPM / 1000 per day / ping / retries) |
| "LLM as judge" | "What if the judge is biased/weak?" (see Q17) |
| "I fixed context_recall" | "What is reference_contexts and where does it come from?" |
| "Streamlit" | "What are Streamlit's limits? Where does session data live?" |
| "ChromaDB" | "Ephemeral or persistent? Why?" |

**Prepared one-liners:**
- "ChromaDB index is ephemeral in-memory, so each session rebuilds it (~30-60s warm-up); that's why you see 'Warming up libraries' first."
- "Results persist to SQLite locally; on the cloud they're session-only, so I committed `sample_results/` as a fallback and the public app is never empty."

---

# 2-MINUTE FINAL PITCH (template)

1. **Problem:** chatbots hallucinate — "before trusting a chatbot, measure its accuracy."
2. **Solution:** a web app — upload document + questions → RAG answers → RAGAS scores, live.
3. **Why free:** Ollama + Groq free API + Streamlit Cloud + FastEmbed = $0, runs local or public.
4. **Top engineering wins (pick 1-2):**
   (a) top_k=5 + overlap=150 fixed the over-refusal "I don't know" bug,
   (b) workers=2 + retries + judge ping handled free-tier rate limits,
   (c) auto-built reference_contexts to enable context_recall scoring.
5. **Outcome:** a user can measure real accuracy on their own document and questions, and compare
   configurations.
6. **Learnings:** LLM-as-judge trade-offs, debugging retrieval vs generation separately, designing
   around API constraints, and measuring before optimising.

---
*Tip: whenever you quote a number, say exactly where it came from ("this run, this experiment,"
"this fix"). Numbers you can point to are ten times more convincing.*