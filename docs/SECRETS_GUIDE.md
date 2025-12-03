# Secrets Management Guide

## ⚠️ CRITICAL: Never Commit Secrets to Git!

**NEVER commit these files to GitHub:**
- ❌ `.env` (contains your actual API keys)
- ❌ `.streamlit/secrets.toml` (contains your actual API keys)
- ✅ `.env.example` (template - safe to commit)
- ✅ `.gitignore` (ensures secrets aren't committed)

---

## How Secrets Work in Production

### The Problem
- Your code needs API keys to work
- But you can't put API keys in your code (security risk)
- And you can't commit `.env` files to GitHub (they'll be public)

### The Solution
**Production platforms provide secure secret management:**
- You set secrets in the platform's dashboard (not in code)
- The platform injects them as environment variables at runtime
- Your code reads from environment variables (already set up!)
- Secrets never appear in your Git repository

---

## Local Development (Your Computer)

### Step 1: Create `.env` file (NOT committed to Git)

```bash
# Copy the template
cp .env.example .env

# Edit .env and add your real API keys
# .env is in .gitignore, so it won't be committed
```

### Step 2: Use `.env` locally

```bash
# Docker automatically reads .env file
docker-compose up

# Or manually:
docker run --env-file .env vajra-prototype
```

**✅ Safe because:**
- `.env` is in `.gitignore`
- Only exists on your local machine
- Never pushed to GitHub

---

## Production Deployment (Cloud Platforms)

### How It Works

1. **You push code to GitHub** (without secrets)
2. **Platform builds your Docker image** (without secrets)
3. **You set secrets in platform dashboard** (separate from code)
4. **Platform injects secrets at runtime** (as environment variables)
5. **Your app reads from environment variables** (already configured!)

### Your Code Already Supports This! ✅

Your `app.py` already checks environment variables:
```python
# First tries Streamlit secrets (for local dev)
# Then falls back to environment variables (for production)
api_key = os.getenv("OPENAI_API_KEY")
```

**This means your code works in both:**
- Local development (reads from `.env` file)
- Production (reads from platform's environment variables)

---

## Setting Secrets on Different Platforms

### Railway

1. Go to your project → **Variables** tab
2. Click **"New Variable"**
3. Add each secret:
   ```
   OPENAI_API_KEY = your-actual-key
   OPENWEATHER_API_KEY = your-actual-key
   GOOGLE_PLACES_API_KEY = your-actual-key
   ```
4. Click **"Save"**
5. Railway automatically redeploys with new secrets

**✅ Secrets are encrypted and never exposed in logs**

### Render

1. Go to your service → **Environment** tab
2. Click **"Add Environment Variable"**
3. Add each secret:
   - Key: `OPENAI_API_KEY`
   - Value: `your-actual-key`
4. Click **"Save Changes"**
5. Render automatically redeploys

**✅ Secrets are encrypted and never exposed in logs**

### Fly.io

```bash
# Set secrets via CLI
fly secrets set OPENAI_API_KEY=your-actual-key
fly secrets set OPENWEATHER_API_KEY=your-actual-key
fly secrets set GOOGLE_PLACES_API_KEY=your-actual-key

# Or set all at once
fly secrets set \
  OPENAI_API_KEY=your-actual-key \
  OPENWEATHER_API_KEY=your-actual-key \
  GOOGLE_PLACES_API_KEY=your-actual-key
```

**✅ Secrets are encrypted and stored securely**

### AWS/GCP/Azure

Each platform has their own secret management:
- **AWS**: Systems Manager Parameter Store, Secrets Manager
- **GCP**: Secret Manager
- **Azure**: Key Vault

Set secrets in their respective dashboards or via CLI.

---

## Best Practices

### ✅ DO:

1. **Use `.env.example` as a template**
   - Shows what secrets are needed
   - Safe to commit to Git
   - Helps other developers know what to set

2. **Set secrets in platform dashboard**
   - Never hardcode in code
   - Never commit to Git
   - Use platform's secret management

3. **Use different keys for dev/prod**
   - Local development: Use test keys
   - Production: Use production keys
   - Limits damage if dev key leaks

4. **Rotate keys regularly**
   - Change API keys periodically
   - Update in platform dashboard
   - No code changes needed

### ❌ DON'T:

1. **Never commit `.env` files**
   - Even if it "works", it's a security risk
   - GitHub history keeps old commits
   - Anyone with repo access can see secrets

2. **Never hardcode secrets in code**
   ```python
   # BAD - Never do this!
   OPENAI_API_KEY = "sk-..."
   ```

3. **Never share secrets in chat/email**
   - Use secure sharing methods
   - Or use platform's team sharing features

4. **Never log secrets**
   - Don't print API keys in logs
   - Don't include in error messages
   - Your code already does this correctly! ✅

---

## Verification Checklist

Before deploying, verify:

- [ ] `.env` is in `.gitignore` ✅
- [ ] `.streamlit/secrets.toml` is in `.gitignore` ✅
- [ ] `.env.example` exists (template) ✅
- [ ] No secrets in code files ✅
- [ ] Secrets set in platform dashboard ✅
- [ ] Code reads from `os.getenv()` ✅ (already done!)

---

## Quick Reference

### Local Development
```bash
# 1. Create .env from template
cp .env.example .env

# 2. Edit .env with your keys
nano .env  # or use your editor

# 3. Run (Docker reads .env automatically)
docker-compose up
```

### Production Deployment
```bash
# 1. Push code (no secrets)
git add .
git commit -m "Deploy"
git push

# 2. Set secrets in platform dashboard
# (Railway/Render/Fly.io - see above)

# 3. Platform deploys with secrets injected
# (Automatic - no code changes needed)
```

---

## Troubleshooting

### "API key not found" in production

1. **Check secrets are set in platform dashboard**
   - Go to your service → Environment/Variables
   - Verify all three keys are present

2. **Check secret names match exactly**
   - Must be: `OPENAI_API_KEY` (not `OPENAI_API_KEY_1`)
   - Case-sensitive!

3. **Redeploy after setting secrets**
   - Some platforms auto-redeploy
   - Others require manual redeploy

4. **Check logs**
   ```bash
   # Railway
   railway logs
   
   # Render
   # View in dashboard
   
   # Fly.io
   fly logs
   ```

### Secrets visible in logs?

- **If you see secrets in logs, rotate them immediately!**
- Check your code doesn't print secrets (yours doesn't ✅)
- Check platform settings for log sanitization

---

## Summary

**Your setup is already secure! ✅**

1. ✅ `.env` is now in `.gitignore` (won't be committed)
2. ✅ `.env.example` exists (template for others)
3. ✅ Code reads from environment variables (works in production)
4. ✅ No secrets hardcoded in code

**For production:**
- Push code to GitHub (no secrets)
- Set secrets in platform dashboard
- Platform injects them at runtime
- Your app reads them automatically

**You're all set!** 🎉

