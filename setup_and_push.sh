#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# Cherenkov EBM — Push to GitHub
# ═══════════════════════════════════════════════════════════════════
# Run this from the cherenkov-ebm directory after you've downloaded
# and unzipped the repo bundle.
#
# Prerequisites:
#   1. You have a GitHub account (psych1cparr0t-dev or whichever name)
#   2. You have git installed and configured with your name/email
#   3. You've created an empty repo on GitHub named "cherenkov-ebm"
#      (Settings → Public → DO NOT initialize with README)
#
# Usage:
#   chmod +x setup_and_push.sh
#   ./setup_and_push.sh <your-github-username>
# ═══════════════════════════════════════════════════════════════════

set -e

USERNAME="${1:-psych1cparr0t-dev}"
REPO_NAME="cherenkov-ebm"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Cherenkov EBM — Repo Setup & Push"
echo "  Target: github.com/${USERNAME}/${REPO_NAME}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Sanity check
if [ ! -f "setup.py" ] || [ ! -d "src/cherenkov" ]; then
    echo "ERROR: Run this script from inside the cherenkov-ebm directory."
    echo "Expected files: setup.py, src/cherenkov/"
    exit 1
fi

# Verify git is configured
if ! git config user.name > /dev/null 2>&1; then
    echo "⚠ Git is not configured. Setting it up now."
    read -p "Your name (e.g., Max Bradford): " GITNAME
    read -p "Your email (e.g., max@cherenkov.industries): " GITEMAIL
    git config --global user.name "$GITNAME"
    git config --global user.email "$GITEMAIL"
fi

echo "[1/5] Initializing git..."
git init -q
git branch -M main

echo "[2/5] Staging files..."
git add .
git status --short | head -30

echo ""
echo "[3/5] Creating initial commit..."
git commit -q -m "Initial release: Cherenkov EBM v4.3.0

- Sklearn-compatible parsing layer with 13-family primitive library
- V1 reinforcement learning module (tabular Q-learning over primitives)
- Validated on 8 synthetic geometric benchmarks (+16.4pp avg)
- Honest real-world results on 5 tabular datasets (consistent with
  navigation-layer framing: no gain expected on linear data)
- Technical preprint v3.0 included
- MIT licensed"

echo ""
echo "[4/5] Adding remote..."
git remote remove origin 2>/dev/null || true
git remote add origin "https://github.com/${USERNAME}/${REPO_NAME}.git"

echo ""
echo "[5/5] Pushing to GitHub..."
echo ""
echo "  About to push to: https://github.com/${USERNAME}/${REPO_NAME}.git"
echo ""
echo "  If you have 2FA enabled, you'll need a Personal Access Token."
echo "  Create one at: https://github.com/settings/tokens"
echo "  Use the token as your password when prompted."
echo ""
read -p "Press Enter to push (or Ctrl+C to abort)..."

git push -u origin main

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✓ Push complete"
echo "  Visit: https://github.com/${USERNAME}/${REPO_NAME}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  Next steps:"
echo "    1. Verify the repo loads at the URL above"
echo "    2. Check that the README renders correctly"
echo "    3. Optional: enable GitHub Pages for docs/"
echo "    4. Send the Wuxi package — GitHub link is now live"
