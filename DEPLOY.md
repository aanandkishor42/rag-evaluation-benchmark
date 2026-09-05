# Deploy the RAG web app publicly (free)

Two free ways. Pick one:

## Option A - Streamlit Cloud + Groq (recommended: $0, 15 minutes)

Groq's free tier (30 req/min, 1,000 req/day, no credit card) replaces the local
Ollama model on the cloud. Embeddings run locally on the cloud CPU via FastEmbed
(free, no API key). There is a second config for this: `config.cloud.yaml`.

1. **Sign up for the free Groq API key:**
   - Groq: https://console.groq.com -> API Keys -> create key (starts with `gsk_`)
2. **Deploy:**
   - Go to https://share.streamlit.io and sign in with your GitHub account
   - "Create app" -> pick your repo `rag-evaluation-benchmark`
   - Main file: `app.py`
   - Click **Deploy**
3. **Add secrets** (after first deploy): on the app page go to Settings -> Secrets and paste:
   ```
   GROQ_API_KEY=gsk_your-key-here
   RAG_CONFIG=config.cloud.yaml
   ```
   (HF_TOKEN is not needed - embeddings run locally with FastEmbed.)
   Then press Rerun.
4. Open your app's URL - it's live, public, and free.

Notes:
- The hosted app answers questions only from uploaded documents (uploaded docs are
  session-only). The full RAGAS benchmark is better run locally (`python run_benchmark.py`)
  because Groq's free tier rate-limits judge calls.
- `dashboard.py` shows whatever is in `results/`; on a fresh cloud deploy that's empty
  until you commit results or run the benchmark.

## Option B - Own small server + local Ollama (visitors unlimited, ~$5/month)

> First test locally (free): make sure the Ollama app is running, then
> `streamlit run app.py` -> upload a file -> Build index -> ask a question.

Pick ONE server:

| Option | Cost | Notes |
| --- | --- | --- |
| Hetzner CX22 (Ubuntu 24.04) | ~$4.50/mo | 2 vCPU, 4 GB RAM - recommended |
| DigitalOcean / Vultr $6 droplet | $6/mo | 2 vCPU, 4 GB RAM |
| Oracle Cloud Always Free | $0 | ARM 4-core/24 GB free tier (setup is fiddly but truly free) |

Requirements: **2 vCPU + 4 GB RAM minimum** (llama3.2:3b + nomic-embed-text need ~4 GB).

## Step B1 - Connect and install Ollama

Open a terminal on your own PC and SSH to the server (replace the IP):

```
ssh root@YOUR_SERVER_IP
```

Then install Ollama and pull the models:

```
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

Check it works:
```
ollama run llama3.2:3b "say hi"
```

## Step B2 - Get the project onto the server

From your own PC:

```
scp -r "C:\Users\Aanand\OneDrive\Desktop\OneDrive\Documents\Default Project" root@YOUR_SERVER_IP:/opt/rag-app
```

Then on the server install Python + the app:

```
sudo apt update && sudo apt install -y python3.12-venv
cd /opt/rag-app
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Step B3 - Start the website

```
.venv/bin/streamlit run app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true
```

Open the server's firewall port 8501 (in your cloud provider's dashboard, allow TCP 8501).
The website is now live at: `http://YOUR_SERVER_IP:8501`

## Step B4 - Keep it running (optional)

Run it in the background so it survives when you close the SSH window:

```
nohup .venv/bin/streamlit run app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true >> app.log 2>&1 &
```

To stop it later: `pkill -f streamlit`

## Step B5 - Nice domain (optional, free)

- Use `http://YOUR_SERVER_IP:8501` directly (works, no TLS), or
- Free DNS: create an account at duckdns.org, put your server IP in, and your site
  becomes `http://yourname.duckdns.org:8501`.

## Notes

- `config.yaml` already points to `provider: ollama`, so no API keys are needed on the server.
- First answer takes a few seconds while Ollama loads the model into memory; the next ones are faster.
- The site keeps no personal data: files stay in the server's memory for the session and are wiped when you stop the app.
- Want the exact same configs tested in your benchmark to be usable in the chat app? Change `chunk_size` / `chunk_overlap` / `top_k` in `config.yaml` - the app reads them from the first experiment at startup.