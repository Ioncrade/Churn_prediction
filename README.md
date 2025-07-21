# Churn Prediction Project

Welcome to the **Churn Prediction** repository! This project delivers a streamlined workflow for analyzing, modeling, and predicting customer churn, featuring step-by-step exploratory data analysis, feature engineering, multiple modeling approaches, and survival analysis. An interactive Streamlit dashboard is also provided for visualization and deployment.

## Table of Contents

- Project Overview
- Features
- Directory Structure
- Installation
- Usage
    - Exploratory Data Analysis (EDA)
    - Feature Engineering
    - Model Training
    - Survival Analysis
    - Streamlit Dashboard
- Images and Visualizations
- Requirements
- Contributing
- License
- Demonstration


## Project Overview

This repository helps businesses efficiently predict and understand customer churn:

- Analyzes raw customer data for churn patterns.
- Engineers features that improve predictive performance.
- Benchmarks standard and advanced models.
- Explores customer duration and churn dynamics using survival analysis.
- Provides interactive, no-code dashboarding with Streamlit.


## Features

- **Deep Exploratory Data Analysis**: Scripts generate visualizations for understanding churn drivers.
- **Flexible Feature Engineering**: Modular approach for adding and testing new features.
- **Baseline \& Advanced Modeling**: Compare machine learning models in a unified framework.
- **Customer Lifetime Analysis**: Use survival analysis to model churn over time.
- **Streamlit Dashboard**: Visualize patterns, test predictions, and interact with the models in real-time.


## Directory Structure

```
Churn_prediction/
│
├── data/                         # Raw and processed datasets
├── images/                       # Visualizations: charts, graphs, EDA outputs
├── .gitignore
├── baseline_models.py            # Baseline machine learning models
├── comprehensive_eda.py          # Full EDA suite
├── eda.py                        # Lightweight EDA scripts
├── feature_engineering.py        # Feature engineering and preprocessing
├── streamlit_app.py              # Streamlit dashboard for interactivity
├── survival_analysis.py          # Time-to-event analysis and plots
```


## Installation

1. **Clone the repository:**

```bash
git clone https://github.com/Ioncrade/Churn_prediction.git
cd Churn_prediction
```

2. **Install dependencies (recommended in a virtual environment):**

```bash
python -m venv venv
source venv/bin/activate    # On Linux/Mac
venv\Scripts\activate       # On Windows

pip install -r requirements.txt
```

## Usage

### Exploratory Data Analysis (EDA)

- Run comprehensive EDA:

```bash
python comprehensive_eda.py
```

- For a quick overview:

```bash
python eda.py
```

**Outputs:**
Charts such as distributions, correlations, and churn patterns are saved to the `images/` directory. Leverage these ready-made images in your reports, presentations, or the Streamlit dashboard.


### Feature Engineering

Transform and select predictive features:

```bash
python feature_engineering.py
```

- Configure or extend transformation steps within the script.


### Model Training

Train machine learning models:

```bash
python baseline_models.py
```

- Output includes metrics like accuracy, ROC curves, and feature importances.


### Survival Analysis

Model time-to-churn using survival curves:

```bash
python survival_analysis.py
```

- Results (plots, tables) are saved to `images/` for further analysis.


### Streamlit Dashboard

Interact with the project through a visual interface:

```bash
streamlit run streamlit_app.py
```

- The dashboard references visuals directly from the `images/` directory, allowing users to browse existing graphs, compare churn rates, and observe model predictions interactively.
- You can also embed these images in presentations or documentation for stakeholders.


## Images and Visualizations

The `images/` directory includes pre-generated charts and visual analytics:

- Distribution of customer features (age, tenure, etc.)
- Churn vs. non-churn histograms
- Correlation heatmaps
- Feature importance rankings
- Survival (time-to-churn) curves
- Model evaluation plots (ROC, confusion matrices, etc.)

Use these visuals to:

- Quickly communicate insights found in exploratory analysis.
- Enhance the Streamlit dashboard by displaying dynamic or summary graphics (loaded from `images/`).
- Provide stakeholders with high-quality plots in documentation and presentations.
- Save time by reusing figures produced during data processing and model training.

Visualizations play an essential role in understanding churn behaviors and should be highlighted in reports, dashboards, and meetings.


## Demonstration
<img width="1522" height="352" alt="Screenshot 2025-07-21 170801" src="https://github.com/user-attachments/assets/c321d0ed-c685-4e56-a1e8-cd8ac656120f" />




