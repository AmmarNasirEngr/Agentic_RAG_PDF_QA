# Deployment Guide

Deploy **AM RAG Document QA** to:

1. **GitHub** — source of truth
2. **HuggingFace Spaces** — free public demo
3. **AWS App Runner** — production, via GitHub Actions CI/CD + Docker

```
                         ┌─────────────────────────┐
   git push main ──────► │        GitHub           │
                         └───────────┬─────────────┘
                                     │
                ┌────────────────────┼─────────────────────┐
                ▼                                            ▼
     ┌────────────────────┐                   ┌──────────────────────────────┐
     │  HuggingFace Space │                   │  GitHub Actions (deploy-aws) │
     │  (Streamlit SDK)   │                   │  build → ECR → App Runner    │
     │  free demo URL     │                   │  https://xxx.awsapprunner.com│
     └────────────────────┘                   └──────────────────────────────┘
```

> **Golden rule:** build and test the Docker image **locally first**, then push. The repo
> ships container-ready from the first commit.

---

## 0. Prerequisites

- Docker Desktop (local build/test)
- Git + a GitHub account
- A HuggingFace account
- An AWS account
- A **DeepSeek API key** (and optional **LangSmith key**)
- `.env` present locally (it is gitignored — never committed)

---

## 1. Build & test the image locally (do this BEFORE pushing)

```bash
docker compose up --build
# open http://localhost:8501  → upload a PDF → ask a question → confirm answer + metrics
```

If this works, the same image will work on App Runner. If it fails here, fix it before
pushing — prod is not where you discover a missing dependency.

---

## 2. Push to GitHub

```bash
git init
git branch -M main
git add .
git status                     # CONFIRM .env is NOT listed
git commit -m "Initial commit: AM RAG QA + Docker + CI/CD"

gh repo create am-rag-qa --public --source=. --remote=origin --push
# or create the repo on github.com and: git remote add origin <url> && git push -u origin main
```

`.gitignore` already excludes `.env`, `uploads/`, `vector_store/`, `storage/`, `results/`.

---

## 3. HuggingFace Spaces (free demo)

The YAML frontmatter at the top of `README.md` configures the Space (`sdk: streamlit`,
`app_file: app.py`).

1. huggingface.co → **New Space** → SDK **Streamlit**.
2. Link the GitHub repo, **or** add the Space as a git remote and push:
   ```bash
   git remote add hf https://huggingface.co/spaces/<user>/am-rag-qa
   git push hf main
   ```
3. Space **Settings → Variables and secrets** → add:
   - `DEEPSEEK_API_KEY` (required)
   - `LANGCHAIN_API_KEY` (optional — enables tracing)
4. The Space builds and goes live at `https://huggingface.co/spaces/<user>/am-rag-qa`.

HF free tier gives 16 GB RAM — comfortable for torch + the embedding model.

---

## 4. AWS App Runner (production)

### 4.1 Create the ECR repository
```bash
aws ecr create-repository --repository-name am-rag-qa --region <AWS_REGION>
```

### 4.2 GitHub OIDC → IAM (no static keys)
1. **IAM → Identity providers → Add provider** → OpenID Connect:
   - URL: `https://token.actions.githubusercontent.com`
   - Audience: `sts.amazonaws.com`
2. Create an IAM role **`am-rag-qa-github-deploy`** with a trust policy limiting it to this
   repo's `main` branch:
   ```json
   {
     "Effect": "Allow",
     "Principal": { "Federated": "arn:aws:iam::<ACCOUNT_ID>:oidc-provider/token.actions.githubusercontent.com" },
     "Action": "sts:AssumeRoleWithWebIdentity",
     "Condition": {
       "StringEquals": { "token.actions.githubusercontent.com:aud": "sts.amazonaws.com" },
       "StringLike": { "token.actions.githubusercontent.com:sub": "repo:<owner>/am-rag-qa:ref:refs/heads/main" }
     }
   }
   ```
3. Attach permissions: ECR push (`ecr:*` on the repo, plus `ecr:GetAuthorizationToken`) and
   App Runner deploy (`apprunner:StartDeployment`, `apprunner:DescribeService`).

### 4.3 App Runner ECR access role
Create role **`am-rag-qa-apprunner-access`** trusted by `build.apprunner.amazonaws.com` with
the `AWSAppRunnerServicePolicyForECRAccess` managed policy (lets App Runner pull from ECR).

### 4.4 First image (manual bootstrap, one time)
App Runner needs an image present before the service can be created. Either push once locally:
```bash
aws ecr get-login-password --region <AWS_REGION> \
  | docker login --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.<AWS_REGION>.amazonaws.com
docker build -t <ACCOUNT_ID>.dkr.ecr.<AWS_REGION>.amazonaws.com/am-rag-qa:latest .
docker push <ACCOUNT_ID>.dkr.ecr.<AWS_REGION>.amazonaws.com/am-rag-qa:latest
```

### 4.5 Create the App Runner service
Console → App Runner → **Create service**:
- **Source:** ECR → `am-rag-qa:latest`; access role = `am-rag-qa-apprunner-access`.
- **Deployment:** Manual (CI triggers it) — or Automatic to deploy on every ECR push.
- **Port:** `8501`
- **Health check:** HTTP path `/_stcore/health`
- **Size (cost-minimized):** **1 vCPU / 2 GB** (2 GB is the floor for torch).
- **Auto-pause:** enabled — App Runner suspends compute when idle, so cost is near-zero
  between visits (first request after idle cold-starts).
- **Environment variables:** `DEEPSEEK_API_KEY` (and optional `LANGCHAIN_API_KEY`). For
  stronger hygiene, store in **AWS Secrets Manager** and reference it.

### 4.6 Wire GitHub Actions
Add these **GitHub repo secrets** (Settings → Secrets and variables → Actions):

| Secret | Value |
|--------|-------|
| `AWS_DEPLOY_ROLE_ARN` | ARN of `am-rag-qa-github-deploy` |
| `AWS_REGION` | e.g. `us-east-1` |
| `ECR_REPOSITORY` | `am-rag-qa` |
| `APP_RUNNER_SERVICE_ARN` | ARN of the App Runner service |

Now every push to `main` runs `.github/workflows/deploy-aws.yml`:
build → push to ECR (`:sha` + `:latest`) → `start-deployment` → wait for `RUNNING`.

---

## 5. CI quality gate

`.github/workflows/ci.yml` runs on every PR to `main`:
- **ruff** lint
- **docker build** (image must build)
- **import smoke test** inside the image (incl. the ragas vertexai shim path)

An optional **RAGAS quality gate** (faithfulness threshold on a fixture PDF) is included
commented-out — enable it by adding `DEEPSEEK_API_KEY` as a repo secret.

---

## 6. Cost & production notes

- **Cost backstop:** set a hard monthly **spend cap on the DeepSeek key** and an **AWS
  Budgets** alert. App Runner auto-pause keeps idle compute cost minimal.
- **Ephemeral storage:** `uploads/`, `vector_store/`, `storage/` reset on redeploy / pause
  recovery — fine for a demo. For persistence: move parents/registry to **S3** and FAISS to
  **EFS** (future work).
- **Image size:** `ragas`/`datasets` stay in `requirements.txt` so the live-scoring toggle
  works at runtime; torch is installed **CPU-only** (Dockerfile `--extra-index-url`) to keep
  the image lean.
- **Secrets:** never commit `.env`. Use Space secrets (HF), env vars / Secrets Manager (AWS).
- **LangSmith:** only active when `LANGCHAIN_API_KEY` is set; the trace URL is hidden in the UI.

---

## 7. Verification checklist

- [ ] `docker compose up` works locally end-to-end
- [ ] GitHub repo pushed; `.env` absent from the repo
- [ ] HF Space live; one query works
- [ ] PR triggers `ci.yml` (lint + build + smoke) green
- [ ] Push to `main` triggers `deploy-aws.yml`; App Runner reaches `RUNNING`
- [ ] App Runner URL serves a working query; `/_stcore/health` returns OK
- [ ] AWS Budgets alert + DeepSeek spend cap configured

---

*Prepared by Ammar Nasir | AI Engineer*
