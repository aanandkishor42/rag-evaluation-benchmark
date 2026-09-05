# Deploy the RAG web app publicly (free for visitors)

This guide turns the local `app.py` (upload documents + chat) into a public website
where anyone can upload their own files and ask questions. Visitors pay nothing -
you run the site on a small server, and the LLM runs free on that server via Ollama.

## Step 0 - Test on your own laptop first (free, 2 minutes)

1. Make sure the Ollama app is running (system tray icon) and models are pulled:
   ```
   ollama pull llama3.2:3b
   ollama pull nomic-embed-text
   ```
2. Start the app:
   ```
   .venv\Scripts\activate
   streamlit run app.py
   ```
3. Your browser opens at `http://localhost:8501`. Upload a file, click
   "Build index", then ask a question.
4. Same-Wi-Fi sharing (free, no server needed):
   ```
   streamlit run app.py --server.address 0.0.0.0 --server.port 8501
   ```
   Others on your network open `http://<YOUR-PC-IP>:8501` (find your IP with `ipconfig`).

## Step 1 - Rent a small server (cheap, ~$5/month - or free!)

Pick ONE of these:

| Option | Cost | Notes |
| --- | --- | --- |
| Hetzner CX22 (Ubuntu 24.04) | ~$4.50/mo | 2 vCPU, 4 GB RAM - recommended |
| DigitalOcean / Vultr $6 droplet | $6/mo | 2 vCPU, 4 GB RAM |
| Oracle Cloud Always Free | $0 | ARM 4-core/24 GB free tier (setup is fiddly but truly free) |

Requirements: **2 vCPU + 4 GB RAM minimum** (llama3.2:3b + nomic-embed-text need ~4 GB).

## Step 2 - Connect and install Ollama

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

## Step 3 - Get the project onto the server

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

## Step 4 - Start the website

```
.venv/bin/streamlit run app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true
```

Open the server's firewall port 8501 (in your cloud provider's dashboard, allow TCP 8501).
The website is now live at: `http://YOUR_SERVER_IP:8501`

## Step 5 - Keep it running (optional)

Run it in the background so it survives when you close the SSH window:

```
nohup .venv/bin/streamlit run app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true >> app.log 2>&1 &
```

To stop it later: `pkill -f streamlit`

## Step 6 - Nice domain (optional, free)

- Use `http://YOUR_SERVER_IP:8501` directly (works, no TLS), or
- Free DNS: create an account at duckdns.org, put your server IP in, and your site
  becomes `http://yourname.duckdns.org:8501`.

## Notes

- `config.yaml` already points to `provider: ollama`, so no API keys are needed on the server.
- First answer takes a few seconds while Ollama loads the model into memory; the next ones are faster.
- The site keeps no personal data: files stay in the server's memory for the session and are wiped when you stop the app.
- Want the exact same configs tested in your benchmark to be usable in the chat app? Change `chunk_size` / `chunk_overlap` / `top_k` in `config.yaml` - the app reads them from the first experiment at startup.