# Deploying Healthcare Chatbot to Streamlit Cloud

This guide walks you through deploying the Medical AI Assistant demo to Streamlit Cloud so recruiters can access it directly.

## Quick Start (5 minutes)

### Step 1: Get a Groq API Key (Free)

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up for a free account
3. Navigate to "API Keys" 
4. Create a new API key
5. Copy and save it securely

### Step 2: Push Code to GitHub

```bash
# If not already a git repo
git init

# Add all files
git add .

# Commit
git commit -m "Add cloud deployment for Streamlit"

# Push to GitHub (create repo on github.com first if needed)
git remote add origin https://github.com/YOUR_USERNAME/Healthcare-Chatbot.git
git push -u origin main
```

### Step 3: Deploy to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **"New app"**
3. Configure:
   - **Repository**: `YOUR_USERNAME/Healthcare-Chatbot`
   - **Branch**: `main`
   - **Main file path**: `app_cloud.py`
4. Click **"Advanced settings"**
5. In the **Secrets** section, add:
   ```toml
   GROQ_API_KEY = "your-actual-groq-api-key"
   ```
6. Click **"Deploy!"**

### Step 4: Generate QR Code

Once deployed, you'll get a URL like: `https://healthcare-chatbot-demo.streamlit.app`

```bash
# Generate QR code with your actual URL
python generate_qr.py https://YOUR-APP-NAME.streamlit.app
```

The QR code will be saved as `demo_qr.png` - add this to your resume!

---

## File Structure for Cloud Deployment

```
Healthcare-Chatbot/
├── app_cloud.py          # Cloud-optimized app (uses Groq API)
├── app.py                # Full local version (uses local LLaMA)
├── requirements_cloud.txt # Lightweight cloud dependencies
├── requirements.txt      # Full dependencies for local
├── .streamlit/
│   ├── config.toml       # Streamlit configuration
│   └── secrets.toml.example  # Example secrets file
├── generate_qr.py        # QR code generator
└── DEPLOY.md             # This file
```

---

## Troubleshooting

### "Module not found" errors
- Make sure Streamlit Cloud is using `requirements_cloud.txt`
- In deploy settings, you can specify a custom requirements file

### API key not working
- Verify the key is active at console.groq.com
- Check that secrets are properly configured in Streamlit Cloud
- Format should be: `GROQ_API_KEY = "gsk_..."`

### App is slow
- First load may take 30-60 seconds as dependencies install
- Subsequent loads are faster

### Memory errors
- The cloud version is optimized for Streamlit Cloud's 1GB limit
- If issues persist, reduce plotly visualizations

---

## Customization

### Update the Medical Knowledge Base

Edit the `MEDICAL_KNOWLEDGE` dictionary in `app_cloud.py` to add more topics:

```python
MEDICAL_KNOWLEDGE = {
    "your_topic": {
        "content": "Information about your topic...",
        "source": "Your Source Name"
    },
    # ... more topics
}
```

### Change the App Name

1. In Streamlit Cloud dashboard, go to your app settings
2. Click the three dots menu
3. Select "Settings" > "General"
4. Update the app URL slug

### Custom Domain (Optional)

You can add a custom domain in Streamlit Cloud settings if you have one.

---

## Resume QR Code Tips

1. **Placement**: Add the QR code in your portfolio/projects section
2. **Size**: Minimum 2cm x 2cm for scanability
3. **Label**: Add "Scan for Live Demo" text near the QR
4. **Test**: Always test the QR code works before printing

### Example Resume Text

```
Healthcare AI Assistant | Python, LangChain, LLaMA 2, Streamlit
- Built RAG-based medical chatbot using LLaMA 2 and FAISS vector search
- Implemented real-time performance analytics with Plotly
- Deployed to cloud with 1-3 second response times
[QR CODE] Scan for Live Demo
```

---

## Support

If you encounter issues:
1. Check [Streamlit Cloud docs](https://docs.streamlit.io/streamlit-community-cloud)
2. Review [Groq API docs](https://console.groq.com/docs)
3. Open an issue on GitHub

---

**Happy Deploying! 🚀**
