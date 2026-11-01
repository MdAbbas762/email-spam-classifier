# START HERE — Email Spam Classifier Upgrade

**Goal:** Update the existing GitHub repo safely, verify it on your computer, then proceed to the Market Leakage Lab project.

## 1) Review what changed

Your original `emails.csv`, `model.pkl`, `vectorizer.pkl`, and historical `email-spam-detection.ipynb` are **preserved**. The notebook has a new warning at the top, and `legacy_app.py` keeps the original CLI source.

The improved `app.py` loads the complete `artifacts/spam_pipeline.joblib` with the same `python app.py` interactive behavior and optional `--text` / `--json`. The project contains a new `spam_classifier/` training/evaluation package, tests, reports, and GitHub CI.

## 2) Verify on Windows

Unzip into a **new folder**, not over your old repository yet.

Open its root folder in VS Code's integrated terminal (PowerShell):

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m spam_classifier.train --data emails.csv --out artifacts
python app.py --text "Hi, our meeting is on Thursday" --json
```

If activation is blocked by PowerShell policy, use `.\.venv\Scripts\python.exe` instead of `python` in each command. If you don't have Python 3.11, use `py -3.12` or your installed supported version.

The training command rebuilds the results. **Do not modify the test set or optimize against its results.**

## 3) Update your existing GitHub repository

After local checks pass:

1. **Make a backup** of your current repository or create a Git branch.
2. Copy the revised contents into your existing cloned repository, keeping its `.git` directory intact. You can also unzip and use `git clone`/copy to avoid losing history. Do not create a new unrelated repository.
3. In the repo, run `git status` and inspect changes. Your existing dataset/model assets are retained; the additional trained model is version-specific.
4. Commit and push:

```powershell
git add .
git commit -m "Add leakage-safe ML evaluation and reproducible tests"
git push origin main
```

If the repository uses a branch name other than `main`, push that branch. If GitHub rejects your push, inspect the message before trying a force-push; **never force-push** as a shortcut.

## 4) Confirm on GitHub

Open the README: charts should render, model comparison should show actual CV results, and GitHub Actions should show passing tests. Then the Email Spam Classifier upgrade is complete.

## 5) Next task

**Market Leakage Lab** — inspect/re-run it on real market data, add analysis worth showcasing, and publish it. The FlowCV tailoring comes after the two projects. No need to edit your resume yet.
