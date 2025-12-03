# Deployment Guide

## GitHub Setup

### 1. Create a New Repository on GitHub

1. Go to [github.com](https://github.com) and sign in
2. Click the "+" icon in the top right → "New repository"
3. Name it: `vajra-prototype` (or your preferred name)
4. Set it to **Public** (required for free Streamlit Cloud)
5. **DO NOT** initialize with README, .gitignore, or license (we already have these)
6. Click "Create repository"

### 2. Push Your Code to GitHub

Run these commands in your terminal:

```bash
cd /Users/anirudhgoyal/Downloads/vajra_prototype

# Add the remote repository (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/vajra-prototype.git

# Push to GitHub
git branch -M main
git push -u origin main
```

If you haven't set up GitHub authentication, you may need to:
- Use a Personal Access Token (Settings → Developer settings → Personal access tokens)
- Or use SSH: `git remote add origin git@github.com:YOUR_USERNAME/vajra-prototype.git`

## Streamlit Cloud Deployment

### 1. Deploy to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click **"New app"**
4. Fill in:
   - **Repository**: Select `YOUR_USERNAME/vajra-prototype`
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - **Python version**: `3.9` or higher
5. Click **"Deploy"**

### 2. Add OpenAI API Key

1. In your Streamlit Cloud app dashboard, click **"Settings"**
2. Scroll down to **"Secrets"**
3. Add your OpenAI API key:
   ```toml
   OPENAI_API_KEY = "sk-your-actual-api-key-here"
   ```
4. Click **"Save"**
5. The app will automatically redeploy

### 3. Access Your Deployed App

Once deployed, you'll get a URL like:
`https://YOUR_APP_NAME.streamlit.app`

Share this URL with anyone who needs access!

## Alternative: Deploy to Other Platforms

### Heroku

1. Create a `Procfile`:
   ```
   web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```
2. Add `runtime.txt`:
   ```
   python-3.9.18
   ```
3. Deploy via Heroku CLI or GitHub integration

### Railway

1. Connect your GitHub repository
2. Railway will auto-detect Python
3. Set environment variable: `OPENAI_API_KEY`
4. Add build command: `pip install -r requirements.txt`
5. Add start command: `streamlit run app.py --server.port=$PORT`

### Docker

Create a `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Then:
```bash
docker build -t vajra-prototype .
docker run -p 8501:8501 -e OPENAI_API_KEY=your-key vajra-prototype
```

