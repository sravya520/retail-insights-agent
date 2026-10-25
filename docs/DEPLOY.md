# Deploying the Retail Insights Agent (free tier)

I cannot deploy this for you. It needs your GitHub account, your Streamlit account and your API key, and this folder has no Git remote yet. This guide has the exact steps. Anything I did not test is marked **untested**.

## What is already prepared

- `app/streamlit_app.py` is the entry point. It runs the agent inside the Streamlit process, so no separate API server is needed.
- `app/requirements.txt` is the requirements file Streamlit Community Cloud reads. It installs CPU-only `torch` to keep the install small.
- The database is rebuilt on first start from `data/deploy/*.parquet` (7.4 MB, committed). The vector index is rebuilt from `data/docs/*.pdf` (committed). Nothing needs to be downloaded by hand, except the embedding model (about 130 MB), which downloads on first start.
- 20 pre-computed answers from real runs are in `data/examples/examples_v1.json` (committed). They open with no quota.
- Live questions are limited: 5 per visitor and 60 per day for the whole demo. Both can be changed with secrets (below).

## Part 1: put the code on GitHub (you do this)

Run these in the project folder. Replace `YOUR-NAME`. The branch here is called `master`. Streamlit accepts any branch name, but `main` is the usual choice.

```bash
git branch -M main
git remote add origin https://github.com/YOUR-NAME/retail-insights-agent.git
git push -u origin main
git push origin --tags        # includes the eval-v1 tag
```

Create the empty repository on github.com first (no README, no license, so the push is clean). Streamlit Community Cloud works with public repositories. The repository does **not** contain `.env`, because it is git-ignored. Check with `git ls-files | grep env` (only `.env.example` should appear).

## Part 2: use a separate API key for the public demo (you do this)

Gemini free quotas are per Google project. If the demo uses the same key as your experiments and the nightly job, a busy day on the demo stops your evals, and the other way round.

1. Open https://aistudio.google.com/apikey.
2. Create a key in a **new project** just for the demo.
3. Keep your current key for experiments and for the GitHub nightly job.

## Part 3: deploy on Streamlit Community Cloud (you do this)

1. Go to https://share.streamlit.io and sign in with GitHub. Allow access to your repository when asked.
2. Click **Create app**, then choose **Deploy a public app from GitHub**.
3. Fill in:
   - Repository: `YOUR-NAME/retail-insights-agent`
   - Branch: `main`
   - Main file path: `app/streamlit_app.py`
4. Open **Advanced settings**:
   - Python version: **3.11**
   - Secrets: paste this, with your demo key:

   ```toml
   GOOGLE_API_KEY = "your-demo-key-here"
   RIA_SESSION_LIMIT = "5"
   RIA_DAILY_LIMIT = "60"
   ```

   Streamlit exposes top-level secrets to the app as environment variables, which is how the app reads them. You can change the limits later in the app settings without a redeploy of the code.
5. Click **Deploy**. The first build installs many packages and can take several minutes.
6. When the app opens, test it in this order:
   1. Open a **pre-computed example** from the sidebar. It should show the answer and the "How this was answered" panel with no quota use.
   2. Ask **one live question**. The first live question after a start builds the vector index and downloads the embedding model, so it can take 1 to 3 minutes. Later ones take about 10 to 25 seconds.
   3. Check that the usage meter in the sidebar goes down by one.
7. Under **Settings, Sharing**, make sure the app is public.
8. Put the app URL into the "Live demo" line at the top of `README.md` and commit.

Things to know:

- Streamlit Community Cloud documents RAM of roughly 690 MB minimum and 2.7 GB maximum (their page says these figures are approximate). I have **not** run this app on Streamlit Cloud. Locally I measured about 624 MB for the agent after the first policy search, plus about 70 MB for Streamlit itself, so roughly 700 MB: close to the documented minimum. If the app shows a "resource limits" error or restarts, use the Docker route.
- A sleeping app wakes when someone opens it (apps sleep after 12 hours without traffic, per the Streamlit docs). The usage counters are kept in a file on the app's disk, so they can reset when the app restarts. The daily limit is a safeguard, not an exact meter.
- Anyone can reach the daily cap of 60 questions. The pre-computed examples keep working when that happens.
- The per-visitor limit uses a hash of the visitor's IP address when the host passes it on, otherwise a random id per browser session. It is a soft limit. A visitor with several browsers can get around it, but the daily cap still holds.

## Part 4: run the nightly eval once by hand on GitHub (you do this)

The workflow `.github/workflows/nightly-eval.yml` already has `workflow_dispatch`, so it can be started manually. I could not run it on GitHub from here (**untested on GitHub**). I did run its exact eval command locally.

1. In the repository, go to **Settings, Secrets and variables, Actions, New repository secret**.
2. Name: `GOOGLE_API_KEY`. Value: your **experiment** key (not the demo key). Save.
3. Go to the **Actions** tab, choose **nightly-eval** on the left, click **Run workflow**, then **Run workflow** again.
4. Wait about 10 to 20 minutes. The first run downloads the dataset (45 MB) and the embedding model.
5. A green run means the offline tests passed, all 10 questions finished and all three gates passed. A red run means a test or a gate failed, or the run stopped early (for example on a free-tier quota). The log says which. The report is attached to the run as an artifact.

It then also runs every night at 01:30 UTC. It uses about 30 model calls a night. GitHub can switch off scheduled workflows in a repository that has had no activity for a long time.

## If Streamlit Community Cloud is not enough (untested alternatives)

- **Hugging Face Spaces (Docker).** Create a Space with the Docker SDK, copy this repository into it, and add `GOOGLE_API_KEY` as a Space secret. The `Dockerfile` listens on 8501 (Streamlit) and 8000 (API); a Space needs the app port set to 8501 in its README front matter (`app_port: 8501`). I did not test this and I did not check the current free hardware limits.
- **Render.** I could not find the free instance RAM on Render's docs page, so I did not rely on it. The agent needs about 700 MB in my local measurement, so I would not expect a 512 MB instance to be enough (Render's free RAM is not on the page I read, so check it yourself).
- **AWS App Runner.** Build the image, push it to Amazon ECR, create an App Runner service from the image on port 8501, and add `GOOGLE_API_KEY` as a secret. App Runner is not free. I did not test this.

## Docker (untested)

```bash
docker build -t retail-insights-agent .
docker run --rm -p 8501:8501 -p 8000:8000 -e GOOGLE_API_KEY=your-key retail-insights-agent
```

Docker Desktop was not running on my machine, so I could not build the image. What I could test is the same start-up path from a clean clone, without Docker (see the next section).

## What I tested instead

A clean clone of the repository was started the way a fresh host would start it (`scripts/start.sh`). The result, the memory measurements and the real error paths are in the Phase 6 section of `docs/results.md`.
