# 🚀 Free Hosting Guide - Streamlit Community Cloud

This guide will help you deploy the Trading Bot Dashboard for **FREE** so your team can access it from anywhere in the world.

## Option 1: Streamlit Community Cloud (RECOMMENDED)

### Why Streamlit Cloud?
- ✅ **100% FREE** - No credit card needed
- ✅ **Auto-deploy** from GitHub
- ✅ **Always online** - 24/7 availability
- ✅ **Global CDN** - Fast access worldwide
- ✅ **Auto-updates** - Push to GitHub = auto deploy

### Step-by-Step Deployment:

#### Step 1: Create GitHub Repository

```powershell
# Navigate to project folder
cd "c:\Users\bajacob\OneDrive - Tecnicas Reunidas, S.A\sandbox\project_2\Trading_bot"

# Initialize Git
git init

# Add all files
git add .

# First commit
git commit -m "Initial commit: Trading Bot Dashboard with Ichimoku Cloud"

# Create repository on GitHub.com first, then:
git remote add origin https://github.com/YOUR_USERNAME/trading-bot-dashboard.git
git branch -M main
git push -u origin main
```

#### Step 2: Deploy to Streamlit Cloud

1. Go to [**share.streamlit.io**](https://share.streamlit.io)
2. Click **"Sign in with GitHub"**
3. Click **"New app"**
4. Fill in:
   - **Repository**: `YOUR_USERNAME/trading-bot-dashboard`
   - **Branch**: `main`
   - **Main file**: `dashboard.py`
5. Click **"Deploy!"**

#### Step 3: Share with Your Team

After deployment (takes 2-3 minutes), you'll get a URL like:
```
https://YOUR_APP_NAME.streamlit.app
```

Share this URL with your team! 🎉

---

## Option 2: Render.com (Alternative)

If Streamlit Cloud doesn't work, use Render.com (also free):

1. Go to [render.com](https://render.com)
2. Connect GitHub
3. Create "Web Service"
4. Select your repo
5. Set build command: `pip install -r requirements.txt`
6. Set start command: `streamlit run dashboard.py --server.port=$PORT --server.address=0.0.0.0`

---

## Option 3: Hugging Face Spaces (Alternative)

Another free option:

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Create new Space
3. Select "Streamlit" SDK
4. Upload your files

---

## ⚠️ Important Notes

### Stock Data
The current app uses local Excel files. For cloud deployment, you have 2 options:

**Option A: Include data in repository** (Easy)
```
# Uncomment this line in .gitignore to INCLUDE stock_data:
# stock_data/
```

**Option B: Use only Alpha Vantage API** (No local files)
- Remove local file option from sidebar
- Use only API-based data loading

### API Keys
For production, use Streamlit Secrets:
1. Go to your app settings on Streamlit Cloud
2. Click "Secrets"
3. Add:
```toml
[alphavantage]
api_key = "YOUR_API_KEY"
```

---

## 📱 Access Modes

| Device | URL |
|--------|-----|
| Desktop | https://YOUR_APP.streamlit.app |
| Mobile | Same URL (responsive design) |
| Tablet | Same URL (responsive design) |

---

## 🔧 Troubleshooting

### "ModuleNotFoundError"
- Check all imports are in `requirements.txt`

### "App not loading"
- Check Streamlit Cloud logs
- Verify `dashboard.py` is in root folder

### "Data not loading"
- Ensure Excel files are included in repo
- Check file paths are relative, not absolute

---

## 📊 Free Tier Limits

### Streamlit Cloud
- **Apps**: Unlimited public apps
- **Resources**: 1GB RAM, shared CPU
- **Sleep**: Apps sleep after inactivity (wake on access)

### Render.com
- **Hours**: 750 free hours/month
- **Resources**: 512MB RAM
- **Sleep**: After 15 min inactivity

---

## Quick Commands Summary

```powershell
# Initialize and push to GitHub
git init
git add .
git commit -m "Trading Bot Dashboard"
git remote add origin https://github.com/YOUR_USERNAME/trading-bot.git
git push -u origin main

# After making changes:
git add .
git commit -m "Updated dashboard"
git push
# Streamlit Cloud auto-deploys in ~1 minute!
```

---

**Your dashboard will be live at:**
`https://trading-bot-dashboard.streamlit.app`

Share this URL with your team! 🌍
