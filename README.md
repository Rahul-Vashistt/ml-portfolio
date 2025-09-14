# ML Portfolio

A collection of machine learning and data science projects demonstrating skills in **EDA, feature engineering, model development, and deployment-ready project structures**. This repo is structured to reflect how ML workflows are handled in real projects, ranging from scratch implementations to applied use cases.

---

## 📂 Repository Structure

<details>
<summary>Click to expand</summary>

```
ml-portfolio/
├── .gitignore
├── EDA/
│   ├── data/
│   │   ├── Tabular/
│   │   │   ├── Batting/
│   │   │   ├── Bowling/
│   │   │   └── Fielding/
│   │   ├── Text_based/
│   │   └── Time_series/
│   └── eda_tabular.ipynb
├── classification/
│   └── iris_flower.ipynb
├── regression/
│   └── house_price_ames/
│       ├── .env
│       ├── config.py
│       ├── data/
│       │   ├── processed/
│       │   ├── raw/
│       │   └── snapshots/
│       ├── models/
│       ├── notebooks/
│       │   └── house_price_pred.ipynb
│       ├── reports/
│       └── src/
│           ├── app/
│           ├── data/
│           ├── features/
│           └── models/
├── scripts/
│   └── make_project_tree.py
├── src/
│   └── viz/
│       ├── __init__.py
│       ├── barplot.py
│       ├── boxplot.py
│       ├── corr_table.py
│       ├── decision_boundary_plot.py
│       ├── heatmap.py
│       ├── histogram.py
│       └── scatterplot.py
├── requirements.txt
├── pyproject.toml
└── README.md
```

</details>

---

## 🚀 Projects

### 📓 Exploratory Data Analysis (EDA)

* **Tabular EDA**: Batting, Bowling, and Fielding datasets
* **Time-series & Text-based EDA**: Structured for expansion
* Notebook: `EDA/eda_tabular.ipynb`

### 📊 Regression

* **House Price Prediction (Ames dataset)**

  * Modular pipeline (`regression/house_price_ames/src/`) for data processing, feature engineering, and modeling
  * Separate folders for raw/processed data, models, and reports
  * Notebook: `regression/house_price_ames/notebooks/house_price_pred.ipynb`

### 🧮 Classification

* **Iris Flower Classification**: Baseline classification workflow

### 🔜 Upcoming Projects

* Customer Churn Predictor
* Titanic Survival Prediction
* Linear Regression (from scratch)
* Image Classification System
* Sentiment Analysis System

---

## 🛠️ Reusable Modules

Located under `src/`:

* **Visualization tools**: barplot, boxplot, heatmap, scatterplot, histograms, correlation tables, decision boundary plots
* Designed to keep plots consistent and reusable across projects

---

## ⚙️ Setup Instructions

Clone the repo:

```bash
git clone https://github.com/<your-username>/ml-portfolio.git
cd ml-portfolio
```

Create a virtual environment:

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📖 Roadmap

* Core ML algorithms from scratch
* Applied ML problems across domains (tabular, text, image, time-series)
* End-to-end pipelines (EDA → feature engineering → modeling → evaluation → reporting)

---

## 📌 Note

This is a **living repository** – more projects and improvements will be added over time. The goal is to build a **production-grade ML portfolio through modular, focused projects**, not just a collection of notebooks.
