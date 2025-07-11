#!/bin/bash

# Set the path to the parent folder
FOLDER="/Users/lyuk/Downloads/Cospectral-Tournament"

# Change to that directory
cd "$FOLDER" || { echo "Folder not found: $FOLDER"; exit 1; }

# Infinite loop to check and push every 10 minutes
while true; do
    echo "[$(date)] Checking for changes in Cospectral-Tournaments..."

    # Stage all changes
    git add .

    # Check if there's anything staged
    if git diff --cached --quiet; then
        echo "No changes to commit."
    else
        git commit -m "Auto-commit: $(date '+%Y-%m-%d %H:%M:%S')"
        git push origin main  # Change 'main' if you're using another branch
        echo "Pushed changes to GitHub."
    fi

    echo "Sleeping for 1 minute..."
    sleep 60
done