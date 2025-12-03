# Container Deployment Guide

This guide will help you deploy the Vajra Prototype as a Docker container and connect it to your custom domain from Namecheap.

## Prerequisites

- Docker installed on your machine ([Install Docker](https://docs.docker.com/get-docker/))
- A domain name registered with Namecheap
- An account on a cloud platform that supports Docker containers and custom domains

## Recommended Platforms

### Option 1: Railway (Recommended - Easiest)
- ✅ Free tier available
- ✅ Easy custom domain setup
- ✅ Automatic HTTPS
- ✅ Simple deployment

### Option 2: Render
- ✅ Free tier available
- ✅ Custom domain support
- ✅ Automatic HTTPS
- ✅ Good documentation

### Option 3: Fly.io
- ✅ Free tier available
- ✅ Global edge network
- ✅ Custom domain support
- ✅ Good for containers

### Option 4: AWS/GCP/Azure
- ✅ Full control
- ✅ Scalable
- ⚠️ More complex setup
- ⚠️ Requires cloud knowledge

---

## Step 1: Build and Test Locally

### 1.1 Build the Docker Image

```bash
cd /Users/anirudhgoyal/Downloads/vajra_prototype
docker build -t vajra-prototype .
```

### 1.2 Test Locally

```bash
# Set your API keys as environment variables
export OPENAI_API_KEY="your-openai-key"
export OPENWEATHER_API_KEY="your-openweather-key"
export GOOGLE_PLACES_API_KEY="your-google-places-key"

# Run the container
docker run -p 8501:8501 \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -e OPENWEATHER_API_KEY=$OPENWEATHER_API_KEY \
  -e GOOGLE_PLACES_API_KEY=$GOOGLE_PLACES_API_KEY \
  vajra-prototype
```

Or use docker-compose:

```bash
# Create a .env file with your API keys
cat > .env << EOF
OPENAI_API_KEY=your-openai-key
OPENWEATHER_API_KEY=your-openweather-key
GOOGLE_PLACES_API_KEY=your-google-places-key
EOF

# Run with docker-compose
docker-compose up
```

Visit `http://localhost:8501` to verify it works.

---

## Step 2: Deploy to Railway

### 2.1 Push to GitHub

1. Create a GitHub repository (if you haven't already)
2. Push your code:

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/vajra-prototype.git
git push -u origin main
```

### 2.2 Deploy on Railway

1. Go to [railway.app](https://railway.app) and sign up/login
2. Click **"New Project"**
3. Select **"Deploy from GitHub repo"**
4. Choose your `vajra-prototype` repository
5. Railway will auto-detect the Dockerfile and start building

### 2.3 Configure Environment Variables

1. In your Railway project, go to **Variables**
2. Add these environment variables:
   ```
   OPENAI_API_KEY=your-openai-key
   OPENWEATHER_API_KEY=your-openweather-key
   GOOGLE_PLACES_API_KEY=your-google-places-key
   ```

### 2.4 Set Up Custom Domain

1. In Railway, go to your service → **Settings** → **Networking**
2. Click **"Generate Domain"** to get a Railway domain first (for testing)
3. Click **"Custom Domain"** → **"Add Custom Domain"**
4. Enter your domain (e.g., `app.yourdomain.com`)
5. Railway will provide DNS records to add

### 2.5 Configure DNS on Namecheap

1. Log into your Namecheap account
2. Go to **Domain List** → Select your domain → **Advanced DNS**
3. Add the DNS records Railway provided:

   **For A Record:**
   - Type: `A Record`
   - Host: `@` (or `app` for subdomain)
   - Value: `[Railway IP address]`
   - TTL: Automatic

   **For CNAME (if provided):**
   - Type: `CNAME Record`
   - Host: `app` (or your subdomain)
   - Value: `[Railway domain].railway.app`
   - TTL: Automatic

4. Wait 5-30 minutes for DNS propagation
5. Railway will automatically provision SSL certificate

---

## Step 3: Deploy to Render (Alternative)

### 3.1 Create Render Account

1. Go to [render.com](https://render.com) and sign up
2. Connect your GitHub account

### 3.2 Create Web Service

1. Click **"New +"** → **"Web Service"**
2. Connect your GitHub repository
3. Configure:
   - **Name**: `vajra-prototype`
   - **Environment**: `Docker`
   - **Region**: Choose closest to you
   - **Branch**: `main`
   - **Dockerfile Path**: `Dockerfile`
   - **Docker Context**: `.` (root)

### 3.3 Add Environment Variables

In **Environment** section, add:
```
OPENAI_API_KEY=your-openai-key
OPENWEATHER_API_KEY=your-openweather-key
GOOGLE_PLACES_API_KEY=your-google-places-key
```

### 3.4 Set Up Custom Domain

1. In your service → **Settings** → **Custom Domains**
2. Click **"Add Custom Domain"**
3. Enter your domain (e.g., `app.yourdomain.com`)
4. Render will provide DNS records

### 3.5 Configure DNS on Namecheap

Add the CNAME record Render provides:
- Type: `CNAME Record`
- Host: `app` (or your subdomain)
- Value: `[Render domain].onrender.com`
- TTL: Automatic

---

## Step 4: Deploy to Fly.io (Alternative)

### 4.1 Install Fly CLI

```bash
# macOS
brew install flyctl

# Or download from https://fly.io/docs/getting-started/installing-flyctl/
```

### 4.2 Login and Create App

```bash
fly auth login
fly launch
```

Follow the prompts to create your app.

### 4.3 Set Secrets

```bash
fly secrets set OPENAI_API_KEY=your-openai-key
fly secrets set OPENWEATHER_API_KEY=your-openweather-key
fly secrets set GOOGLE_PLACES_API_KEY=your-google-places-key
```

### 4.4 Deploy

```bash
fly deploy
```

### 4.5 Add Custom Domain

```bash
fly domains add app.yourdomain.com
```

Fly.io will provide DNS records. Add them to Namecheap.

---

## Step 5: Verify Deployment

1. Wait for DNS propagation (5-30 minutes, sometimes up to 48 hours)
2. Check DNS propagation: [whatsmydns.net](https://www.whatsmydns.net)
3. Visit your custom domain: `https://app.yourdomain.com`
4. Test the application functionality

---

## Troubleshooting

### Container won't start
- Check logs: `docker logs <container-id>`
- Verify environment variables are set
- Check port 8501 is exposed

### DNS not resolving
- Wait longer (up to 48 hours)
- Verify DNS records are correct in Namecheap
- Check DNS propagation status
- Clear your DNS cache: `sudo dscacheutil -flushcache` (macOS)

### SSL certificate issues
- Most platforms auto-provision SSL
- Ensure DNS is correctly configured first
- Wait for certificate provisioning (can take a few minutes)

### App loads but API calls fail
- Verify all API keys are set correctly
- Check environment variables in your platform's dashboard
- Review application logs for errors

---

## Production Considerations

### Security
- ✅ Never commit API keys to Git
- ✅ Use environment variables for all secrets
- ✅ Enable HTTPS (automatic on most platforms)
- ✅ Consider adding authentication if needed

### Performance
- Consider upgrading from free tier for production
- Monitor resource usage
- Set up health checks (already in Dockerfile)

### Database Persistence
- The SQLite database will be ephemeral unless you add volume mounts
- Consider upgrading to PostgreSQL/MySQL for production
- Or use a managed database service

### Monitoring
- Set up error tracking (Sentry, etc.)
- Monitor uptime
- Track API usage and costs

---

## Quick Reference

### Build locally
```bash
docker build -t vajra-prototype .
docker run -p 8501:8501 -e OPENAI_API_KEY=xxx vajra-prototype
```

### Update deployment
```bash
git add .
git commit -m "Update app"
git push
# Platform will auto-redeploy
```

### View logs
- Railway: Project → Deployments → View logs
- Render: Service → Logs
- Fly.io: `fly logs`

---

## Support

If you encounter issues:
1. Check platform-specific documentation
2. Review container logs
3. Verify DNS configuration
4. Test locally first

Good luck with your deployment! 🚀

