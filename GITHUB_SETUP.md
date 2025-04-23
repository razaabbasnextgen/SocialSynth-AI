# GitHub Setup Instructions

Follow these steps to push the SocialSynth-AI project to your GitHub repository:

## Prerequisites

1. [Create a GitHub account](https://github.com/join) if you don't already have one
2. [Install Git](https://git-scm.com/downloads) on your local machine
3. Configure Git with your credentials:
   ```bash
   git config --global user.name "Raza Abbas"
   git config --global user.email "razaabbas2529@gmail.com"
   ```

## Steps to Push to GitHub

1. Create a new repository on GitHub:
   - Go to https://github.com/new
   - Name it "SocialSynth-AI"
   - Set it to Public or Private as desired
   - Do not initialize with README, .gitignore, or license (we already have these)
   - Click "Create repository"

2. Initialize local repository (run these commands in the project root):
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   ```

3. Connect to the remote repository (replace `razaabbasnextgen` with your GitHub username):
   ```bash
   git remote add origin https://github.com/razaabbasnextgen/SocialSynth-AI.git
   ```

4. Push the code to GitHub:
   ```bash
   git push -u origin main
   ```
   Note: If your default branch is "master" instead of "main", use:
   ```bash
   git push -u origin master
   ```

## Additional Information

- The `.gitignore` file is already set up to exclude sensitive files like `.env` and virtual environment directories
- Make sure not to commit any API keys or sensitive information
- For collaborators, they can clone the repository using:
  ```bash
  git clone https://github.com/razaabbasnextgen/SocialSynth-AI.git
  ```

## Maintaining the Repository

- Add new features with:
  ```bash
  git checkout -b new-feature
  # Make changes
  git add .
  git commit -m "Add new feature"
  git push origin new-feature
  ```
- Create a pull request on GitHub to merge changes
- Keep your local repository updated:
  ```bash
  git pull origin main
  ``` 