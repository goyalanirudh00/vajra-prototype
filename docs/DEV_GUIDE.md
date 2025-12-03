# Development Guide - Testing Changes with Docker

## Quick Start for Development

### Option 1: Development Mode (Recommended - Auto-reloads on changes)

```bash
# Stop any running containers
docker-compose down

# Start in development mode (with file watching)
docker-compose -f docker-compose.dev.yml up
```

**What this does:**
- Mounts your code files as volumes (changes reflect immediately)
- Streamlit automatically detects file changes and reloads
- You can edit code and see changes in seconds!

### Option 2: Production Mode (Rebuild required)

```bash
# Stop container
docker-compose down

# Rebuild and restart
docker-compose up -d --build
```

**When to use:**
- When you've made significant changes
- When you've updated `requirements.txt`
- For final testing before deployment

---

## How It Works

### Development Mode (docker-compose.dev.yml)

1. **Code Mounting**: Your Python files are mounted as volumes
   - `app.py` → `/app/app.py` in container
   - `metrics.py` → `/app/metrics.py` in container
   - `weather_api.py` → `/app/weather_api.py` in container

2. **Auto-Reload**: Streamlit watches for file changes
   - Edit a file → Save → Streamlit detects change → Reloads automatically
   - Usually takes 2-5 seconds to see changes

3. **No Rebuild Needed**: Since code is mounted, you don't need to rebuild the image

### Production Mode (docker-compose.yml)

1. **Code is Copied**: Files are copied into the image during build
2. **Changes Require Rebuild**: Must rebuild image to see changes
3. **Use for**: Final testing, deployment

---

## Common Workflows

### Making Code Changes

```bash
# 1. Start in dev mode (if not already running)
docker-compose -f docker-compose.dev.yml up

# 2. Edit your code files (app.py, metrics.py, etc.)
# 3. Save the file
# 4. Wait 2-5 seconds - Streamlit will auto-reload!
# 5. Refresh your browser to see changes
```

### Adding New Python Packages

```bash
# 1. Edit requirements.txt
# 2. Rebuild the image (packages need to be installed)
docker-compose down
docker-compose -f docker-compose.dev.yml build
docker-compose -f docker-compose.dev.yml up
```

### Viewing Logs While Developing

```bash
# In a separate terminal, watch logs
docker logs -f vajra-prototype-dev
```

### Stopping Development Server

```bash
# Press Ctrl+C in the terminal, or:
docker-compose -f docker-compose.dev.yml down
```

---

## Troubleshooting

### Changes Not Reflecting?

1. **Check if file is mounted correctly:**
   ```bash
   docker exec vajra-prototype-dev ls -la /app/app.py
   ```

2. **Check Streamlit logs:**
   ```bash
   docker logs vajra-prototype-dev | tail -20
   ```

3. **Force reload:** Refresh your browser (Cmd+R or Ctrl+R)

4. **Restart container:**
   ```bash
   docker-compose -f docker-compose.dev.yml restart
   ```

### Streamlit Not Detecting Changes?

- Make sure you're using `docker-compose.dev.yml` (dev mode)
- Check that files are actually saved
- Try manually triggering reload: Click the "Always rerun" button in Streamlit UI

### Import Errors After Changes?

- If you added new imports, make sure packages are in `requirements.txt`
- Rebuild the image: `docker-compose -f docker-compose.dev.yml build`

---

## Tips

1. **Keep dev mode running**: Leave `docker-compose -f docker-compose.dev.yml up` running in a terminal while you code

2. **Watch logs**: Open a second terminal and run `docker logs -f vajra-prototype-dev` to see errors immediately

3. **Browser auto-refresh**: Some browsers can auto-refresh when Streamlit reloads

4. **Test before deploying**: Always test in production mode (`docker-compose up`) before deploying to ensure everything works

---

## Quick Reference

```bash
# Start dev mode
docker-compose -f docker-compose.dev.yml up

# Start dev mode in background
docker-compose -f docker-compose.dev.yml up -d

# View logs
docker logs -f vajra-prototype-dev

# Stop dev mode
docker-compose -f docker-compose.dev.yml down

# Rebuild after requirements.txt changes
docker-compose -f docker-compose.dev.yml build
docker-compose -f docker-compose.dev.yml up
```

