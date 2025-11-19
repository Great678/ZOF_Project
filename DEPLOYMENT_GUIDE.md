# ZOF Solver - Complete Deployment Guide

## Quick Start (Local Testing)

### 1. Install Dependencies

```bash
cd /Users/HP/Downloads/ZOF
pip install -r requirements.txt
```

### 2. Test CLI Application

```bash
python ZOF_CLI.py
```

### 3. Test Web Application

```bash
python app.py
```

Then open: http://localhost:5000

### 4. Run Tests

```bash
python test_solver.py
```

---

## Deployment Options

### Option 1: Render.com (RECOMMENDED)

#### Step-by-Step Instructions:

1. **Prepare GitHub Repository**

   ```bash
   cd /Users/HP/Downloads/ZOF
   git init
   git add .
   git commit -m "Initial commit: ZOF Solver project"
   ```

2. **Create GitHub Repository**

   - Go to https://github.com/new
   - Create new repository named "ZOF-Solver"
   - Don't initialize with README (we already have one)

3. **Push to GitHub**

   ```bash
   git remote add origin https://github.com/YOUR-USERNAME/ZOF-Solver.git
   git branch -M main
   git push -u origin main
   ```

4. **Deploy on Render**

   - Go to https://render.com
   - Sign up/Login (can use GitHub account)
   - Click "New +" → "Web Service"
   - Select "Build and deploy from a Git repository"
   - Connect your GitHub account if not connected
   - Select your "ZOF-Solver" repository
   - Configure:
     - Name: `zof-solver` (or any name you prefer)
     - Region: Choose closest to you
     - Branch: `main`
     - Root Directory: (leave blank)
     - Runtime: `Python 3`
     - Build Command: `pip install -r requirements.txt`
     - Start Command: `gunicorn app:app`
     - Instance Type: Free
   - Click "Create Web Service"
   - Wait 5-10 minutes for deployment
   - Copy the URL (e.g., https://zof-solver.onrender.com)

5. **Update ZOF_hosted_webGUI_link.txt**
   - Add your live URL to the file

---

### Option 2: PythonAnywhere

#### Step-by-Step Instructions:

1. **Sign Up**

   - Go to https://www.pythonanywhere.com
   - Create free account

2. **Upload Files**

   - Go to "Files" tab
   - Create directory: `ZOF`
   - Upload all your project files

3. **Install Dependencies**

   - Go to "Consoles" tab
   - Start a Bash console

   ```bash
   cd ZOF
   pip install --user -r requirements.txt
   ```

4. **Create Web App**

   - Go to "Web" tab
   - Click "Add a new web app"
   - Choose "Manual configuration"
   - Select Python 3.10

5. **Configure WSGI File**

   - Click on WSGI configuration file link
   - Replace content with:

   ```python
   import sys
   path = '/home/YOURUSERNAME/ZOF'
   if path not in sys.path:
       sys.path.append(path)

   from app import app as application
   ```

6. **Set Working Directory**

   - In Web tab, set working directory to: `/home/YOURUSERNAME/ZOF`

7. **Reload**
   - Click green "Reload" button
   - Your app will be at: https://YOURUSERNAME.pythonanywhere.com

---

### Option 3: Streamlit Cloud (Alternative - Requires Streamlit Version)

#### Note: This requires converting to Streamlit app

1. **Create Streamlit Version** (Optional)

   - Create `streamlit_app.py` instead of Flask app
   - Use Streamlit components

2. **Deploy**
   - Go to https://streamlit.io/cloud
   - Connect GitHub repository
   - Deploy main Streamlit file

---

### Option 4: Railway.app

#### Step-by-Step Instructions:

1. **Sign Up**

   - Go to https://railway.app
   - Sign up with GitHub

2. **New Project**

   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Select your ZOF-Solver repository

3. **Configure**

   - Railway auto-detects Python
   - Set start command: `gunicorn app:app --bind 0.0.0.0:$PORT`
   - Deploy automatically

4. **Get URL**
   - Go to Settings → Domains
   - Generate domain
   - Copy URL

---

## Post-Deployment Checklist

- [ ] Web app loads successfully
- [ ] All 6 methods are accessible
- [ ] Can enter functions and parameters
- [ ] Results display correctly
- [ ] Iteration tables show data
- [ ] No console errors
- [ ] Works on mobile devices
- [ ] Updated ZOF_hosted_webGUI_link.txt with:
  - Your name
  - Matric number
  - Live URL
  - GitHub repository link

---

## Testing Your Deployed App

### Test Cases:

1. **Bisection Method**

   - Function: `x**2 - 4`
   - a = 0, b = 3
   - Expected root: ≈ 2.0

2. **Newton-Raphson**

   - Function: `x**3 - 2*x - 5`
   - x0 = 2
   - Expected root: ≈ 2.094551

3. **Secant Method**

   - Function: `cos(x) - x`
   - x0 = 0, x1 = 1
   - Expected root: ≈ 0.739085

4. **Fixed Point**
   - Function: `sqrt(x + 2)`
   - x0 = 1
   - Expected root: ≈ 2.0

---

## Troubleshooting

### Common Issues:

1. **App won't start**

   - Check requirements.txt has all dependencies
   - Verify Procfile is correct
   - Check logs on hosting platform

2. **Module not found errors**

   - Ensure requirements.txt includes all packages
   - Try: `pip freeze > requirements.txt`

3. **Port binding issues**

   - Use: `app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))`

4. **Math function errors**
   - Ensure sympy is installed
   - Check function syntax

### Debug Mode (Local Only):

```python
# In app.py, for local testing only:
if __name__ == '__main__':
    app.run(debug=True)
```

**IMPORTANT**: Never deploy with debug=True in production!

---

## Final Submission

### GitHub Repository Structure:

```
ZOF/
├── README.md
├── requirements.txt
├── Procfile
├── .gitignore
├── ZOF_CLI.py
├── app.py
├── test_solver.py
├── ZOF_hosted_webGUI_link.txt
├── templates/
│   └── index.html
└── static/
    └── style.css
```

### Submission to Scorac.com:

1. Ensure all files are in ZOF folder
2. Compress folder to ZIP
3. Submit to Scorac.com
4. Include link to GitHub repository
5. Include live deployment URL

---

## Support Contacts

- Render Support: https://render.com/docs
- PythonAnywhere Help: https://help.pythonanywhere.com
- GitHub Issues: Create issue in your repository

---

## Additional Resources

- Flask Documentation: https://flask.palletsprojects.com/
- SymPy Documentation: https://docs.sympy.org/
- Numerical Methods Reference: https://en.wikipedia.org/wiki/Root-finding_algorithms

---

**Good luck with your deployment! 🚀**
