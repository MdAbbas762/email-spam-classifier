# Dashboard screenshots — capture your actual running project

These screenshots are optional, but useful for the repository's GitHub landing page. Do not use simulated screenshots or make performance claims that differ from your local `metrics.json`.

## 1. Run the interface

From the repository root in Windows PowerShell or Command Prompt:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-ui.txt
.\.venv\Scripts\python.exe -m streamlit run dashboard.py
```

Open the local address shown in the terminal (typically `http://localhost:8501`). Keep the terminal running while capturing screenshots.

## 2. Suggested screenshots

1. **`screenshots/01-message-prediction.png`** — Select the Promotional spam sample, click **Analyze Message**, and show the prediction plus top contributing terms.
2. **`screenshots/02-evaluation.png`** — Open **Evaluation** and capture the four metrics with the confusion matrix.
3. **`screenshots/03-model-comparison.png`** — Capture the cross-validation chart and metric table under **Evaluation**.
4. **`screenshots/04-batch-analysis.png`** — Optionally upload a small CSV of synthetic example messages, classify it, and capture the results.

Use a readable browser window and 100% zoom. Do not include private client emails or personal inbox data. The demo messages supplied in the UI are fictional.

To capture on Windows: press **Win + Shift + S**, select the relevant dashboard area, and save each PNG in a new `screenshots/` folder at the repository root.

## 3. Add real screenshots to GitHub README

After capturing the first two images, insert the following markdown near the top of `README.md`, immediately after the opening project description:

```markdown
## 🖥️ Application Preview

### Live Message Classification
![Message Classification](screenshots/01-message-prediction.png)

### Holdout Model Evaluation
![Model Evaluation](screenshots/02-evaluation.png)
```

Only add the image links **after** the PNG files exist in the repository. Commit the screenshots and README update together.
