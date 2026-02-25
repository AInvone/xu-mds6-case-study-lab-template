# MDS6 – Cognitive Computing: 
## Machine Learning Lab Template (Case Study Scaffold)

This repository provides a structured lab scaffold to support the **case study examination** in **MDS6 – Cognitive Computing**.

It supports both: 
- the guided **lab work**, and 
- the assessed **case study projec**. 

The template is designed to enforce good experimental practice:
- clear problem formulation 
- correct separation of train/validation/test usage
- baseline-first thinking
- cross-validation for robust model selection
- reproducibility and methodological transparency (fixed seeds, documented dependencies, rerunnable pipeline)

You may use **Google Colab** or run locally/on your own infrastructure.
Your submission must be reproducible in a clean environment.

---

## Purpose of this Repository

This repository is **not a finished solution**.

It is a structured scaffold that ensures:

- Correct experimental design
- Avoidance of data leakage
- Transparent evaluation
- Reproducibility of results

Students are expected to extend and adapt this scaffold for their case study.

---

## Available Case Study Options

Students choose **one** of the following datasets. 

### Option A — Telco Customer Churn (Classification)
- Objective: Predict whether a customer will churn. 
- Type: Binary classification.
- Dataset: Public IBM Telco dataset (downloaded automatically from a public source (IBM GitHub raw CSV) on first run.

Focus areas:
- Class imbalance
- Metric selection (Accuracy vs. Precision vs. Recall vs. F1 vs. ROC-AUC)
- Error analysis 

--- 

### Option B — California Housing (Regression)

- Objective: Predict median house value. 
- Type: Regression.
- Dataset: Loaded via `sklearn.datasets.fetch_california_housing()`.

Focus areas:
- MAE vs RMSE vs R2
- Residual analysis
- Model bias vs variance 

---

## Repository Structure

```
.
├── notebooks/
│   ├── 01_telco_case_study.ipynb
│   └── 02_housing_case_study.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loading.py
│   ├── preprocessing.py
│   ├── splitting.py
│   ├── baseline.py
│   ├── metrics.py
│   └── evaluation.py
├── requirements.txt
└── README.md

``` 

---

## Core Experimental Protocol (mandatory)

To ensure methodological correctness:

**Goal:** estimate generalization performance correctly.

1. Perform a **train/test split once at the beginning**.
2. Use **cross-validation only on the training set** for model comparison / selection / hyperparameter tuning.
3. Select the best-performing model configuration based on CV.
4. **Retrain** the selected model on the **full training dataset**.
5. Evaluate performance **exactly once** on the **untouched test set**.
6. Report and justify metric choice.

> **Do not** use the test set for model selection or tuning.  
> If you tune after seeing test results, your test set becomes training data (invalid evaluation).

Violating this principle invalidates the evaluation.

---

## Baseline Requirement

Before implementing complex models, a **baseline model must be evaluated**:

- Classification: Majority class predictor  
- Regression: Mean predictor  

All advanced models must be compared against this baseline using the same evaluation protocol.

---

## Reproducibility Requirements

Submissions must be reproducible.

This includes:

- Fixed random seed (`RANDOM_STATE`)
- Deterministic train/test split
- Clearly specified dependencies (`requirements.txt`)
- Executable notebook from start to finish
- Reported metrics must match code output

If the code cannot be executed or results cannot be reproduced, the submission cannot be graded.

---

## Setting up the environment

Set up a local environment once so you can install the packages from `requirements.txt` and run the notebooks smoothly.

### Prerequisites

- **Python 3.10 or 3.11** (recommended). Check with: `python3 --version`
- **pip** (usually included with Python). Check with: `pip --version`

### Step 1 — Clone or download the repository

```bash
cd /path/to/your/projects
git clone <repository-url>
cd xu-mds6-case-study-lab-template
```

Or download and unzip the repo, then open a terminal in the project root (the folder that contains `src/`, `requirements.txt`, and the notebooks).

### Step 2 — Create and activate a virtual environment

Using a virtual environment keeps this project’s dependencies separate from your system Python.

**On macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```
## Remark: .venv is the name you choose for this environment, you can also choose some other name for it

**On Windows (Command Prompt):**

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

**On Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

You should see `(.venv)` (or similar) in your prompt. The rest of the steps assume this environment is activated.

**Optional — using Conda:**

```bash
conda create -n mds6-lab python=3.11 -y
conda activate mds6-lab
```

### Step 3 — Install dependencies

From the **project root** (same directory as `requirements.txt`):

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs: `numpy`, `pandas`, `scikit-learn`, and `matplotlib` (and their dependencies).

### Step 4 — Verify the installation

Check that the main packages are available:

```bash
python -c "import numpy, pandas, sklearn, matplotlib; print('All packages OK')"
```

If you see `All packages OK`, the environment is ready.

### Step 5 — Run the notebooks

- **Jupyter:** From the project root, run `jupyter notebook` or `jupyter lab`, then open `01_telco_case_study.ipynb` or `02_housing_case_study.ipynb`. The notebook’s working directory will be the project root, so `from src.data_loading import ...` will work.
- **VS Code / Cursor:** Open the project folder (the repo root), then open a notebook. Select the kernel that uses your `.venv` (e.g. “Python 3.x.x (’.venv’)”). Run cells from top to bottom.
- **Command line (headless):** `jupyter nbconvert --execute --to notebook 01_telco_case_study.ipynb` (run from project root).

**Important:** Always run notebooks from the **project root** so that the `src/` package can be imported.

---

## Running the Project

### Option 1 — Google Colab

1. Open the notebook in Google Colab (from GitHub or by uploading the repo ZIP).
2. Ensure the **entire repo** (including the `src/` folder with the `.py` files) is available in Colab so that `from src.data_loading import ...` works. After uploading a ZIP, unzip it and set Colab’s working directory to the repo root if needed.
3. Install dependencies:
   - Run the first cell: `pip install -r requirements.txt`

   Or: 
   ```python
   !pip install -r ../requirements.txt
   ```
   (adjust the path to `requirements.txt` if your notebook is in a subfolder.)
4. Run the notebook from top to bottom.

**Notes:**

- The Telco dataset is automatically downloaded into `./data/` if not present when running `load_telco()`.
- The Housing dataset is fetched by scikit-learn and cached automatically.

---

### Option 2 — Local Environment

1. Set up the environment following the section [**Setting up the environment**](#setting-up-the-environment) above (create venv, install from `requirements.txt`).
2. Open and run the notebook from top to bottom using Jupyter, VS Code, or similar. Ensure the notebook is run from the **project root** so `from src.data_loading import ...` works.

--- 

## Case Study Submission

Students must submit a ZIP archive containing:
The completed notebook (Telco or Housing)

- All modified or added src/ files
- requirements.txt
- A short README section explaining how to run the project

The submission must run end-to-end in a clean environment.

--- 

## Academic Integrity

- All modeling decisions must be justified.
- External sources (if used) must be properly referenced.
- Code must be written by the submitting group.
- Superficial modifications of baseline code are insufficient.

--- 
## Final Remarks

Machine learning is not model selection.
It is experimental design under uncertainty.

This template is designed to support rigorous, reproducible, and interpretable machine learning practice.