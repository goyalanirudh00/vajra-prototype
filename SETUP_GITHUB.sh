#!/bin/bash

# Script to set up GitHub repository for Vajra Prototype
# Run this after creating a repository on GitHub

echo "🚀 Setting up GitHub repository for Vajra Prototype"
echo ""

# Check if remote already exists
if git remote get-url origin &> /dev/null; then
    echo "⚠️  Remote 'origin' already exists:"
    git remote -v
    echo ""
    read -p "Do you want to update it? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        read -p "Enter your GitHub username: " GITHUB_USERNAME
        read -p "Enter your repository name (default: vajra-prototype): " REPO_NAME
        REPO_NAME=${REPO_NAME:-vajra-prototype}
        git remote set-url origin "https://github.com/${GITHUB_USERNAME}/${REPO_NAME}.git"
        echo "✅ Updated remote to: https://github.com/${GITHUB_USERNAME}/${REPO_NAME}.git"
    fi
else
    read -p "Enter your GitHub username: " GITHUB_USERNAME
    read -p "Enter your repository name (default: vajra-prototype): " REPO_NAME
    REPO_NAME=${REPO_NAME:-vajra-prototype}
    
    echo ""
    echo "📝 Make sure you've created the repository on GitHub first!"
    echo "   Go to: https://github.com/new"
    echo "   Repository name: ${REPO_NAME}"
    echo "   Set to: Public"
    echo "   DO NOT initialize with README, .gitignore, or license"
    echo ""
    read -p "Press Enter once you've created the repository..."
    
    git remote add origin "https://github.com/${GITHUB_USERNAME}/${REPO_NAME}.git"
    echo "✅ Added remote: https://github.com/${GITHUB_USERNAME}/${REPO_NAME}.git"
fi

echo ""
echo "📤 Pushing to GitHub..."
git branch -M main
git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Successfully pushed to GitHub!"
    echo ""
    echo "🌐 Next steps:"
    echo "   1. Go to https://share.streamlit.io"
    echo "   2. Sign in with GitHub"
    echo "   3. Click 'New app'"
    echo "   4. Select your repository: ${GITHUB_USERNAME}/${REPO_NAME}"
    echo "   5. Set main file: app.py"
    echo "   6. Add your OPENAI_API_KEY in Secrets"
    echo "   7. Deploy!"
else
    echo ""
    echo "❌ Push failed. You may need to:"
    echo "   - Set up GitHub authentication (Personal Access Token)"
    echo "   - Or use SSH: git remote set-url origin git@github.com:${GITHUB_USERNAME}/${REPO_NAME}.git"
fi

