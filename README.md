# 🌙 SleepSync AI

> **Interactive sleep & productivity analytics dashboard** — visualizes the relationship between sleep patterns, work habits, and wellness across professions using machine learning and feature engineering.

🌐 **Live Demo:** [sleepsync.streamlit.app](https://sleepsync.streamlit.app/)
&nbsp;&nbsp;|&nbsp;&nbsp;
📁 **GitHub:** [SleepSync](https://github.com/Chirudeep2002/SleepSync-Visual-Analytics-Machine-Learning-for-Productivity-Prediction)

---

## 📊 Project Stats

| Metric | Value |
|---|---|
| 📦 Dataset | **500** records across **5 professions** |
| 👔 Professions Covered | Artist · Doctor · Engineer · Teacher · Tech Worker |
| 🔬 ML Models Deployed | 3 (Linear Regression · Random Forest · XGBoost) |
| ⚙️ Engineered Features | Sleep Efficiency · Recovery Score · Work-Life Balance Index · Sleep Debt |
| 📊 Visualizations | Heatmaps · Scatter plots · Box plots · Histograms · Bar charts |
| 🚀 Deployment | Streamlit Community Cloud |

---

## 🧠 What It Does

SleepSync is a data-driven wellness dashboard that allows users to interactively explore how sleep duration, sleep debt, and work hours relate to productivity and burnout risk across different professional groups. It provides a hands-on ML experimentation environment — users can train and compare three models live, inspect feature importance, and generate personalized wellness insights.

The platform is designed to demonstrate the full ML workflow: raw data → feature engineering → model training → evaluation → interactive prediction.

---

## 🚀 Features

### 📊 Interactive Data Analytics
- Dynamic profession-level filtering across 5 career groups
- Correlation heatmaps: sleep duration, work hours, sleep debt, productivity
- Histogram, scatter plot, box plot, and bar chart views
- KPI cards: average sleep duration, productivity score, burnout risk index

### ⚙️ Feature Engineering Pipeline
Four domain-informed features engineered from raw inputs:

| Feature | Formula |
|---|---|
| **Sleep Efficiency** | `Sleep Duration / 9` (normalized to optimal 9-hr baseline) |
| **Sleep Debt** | `8 - Sleep Duration` (cumulative deficit from target) |
| **Recovery Score** | `max(0, 1 - Sleep Debt / 4)` (wellness recovery index) |
| **Work-Life Balance** | `Sleep Duration / Work Hours` (balance ratio) |

### 🤖 Live ML Experimentation
Train and compare three models directly in the browser:
- **Linear Regression** — baseline interpretability
- **Random Forest Regressor** — ensemble, handles non-linearity
- **XGBoost Regressor** — gradient boosting with feature importance

Outputs per model: MAE · MSE · RMSE · R² Score · Feature Importance chart

### 🧠 AI Productivity Prediction
Input your own sleep and work parameters to receive:
- Predicted productivity level
- Estimated burnout risk score
- Sleep health score
- AI-generated wellness recommendations

### 🌡️ Wellness Insights Module
- Profession-level productivity benchmarking
- Burnout detection based on sleep debt accumulation
- Personalized sleep health recommendations
- Work-life balance scoring

---

## 🛠️ Tech Stack

| Layer | Technologies |
|---|---|
| **ML** | Scikit-learn · XGBoost · Pandas · NumPy |
| **Visualization** | Plotly (interactive charts) |
| **Frontend** | Streamlit · Custom CSS (dark theme) |
| **Deployment** | Streamlit Community Cloud |

---

## 📂 Project Structure

```
SleepSync-AI/
│
├── app.py                          # Main Streamlit app
├── requirements.txt
├── README.md
├── sleep_productivity_dataset.csv  # Primary dataset (500 records)
└── sleep_productivity_dataset-2.csv
```

---

## ⚙️ Setup & Installation

```bash
# 1. Clone the repository
git clone https://github.com/Chirudeep2002/SleepSync-Visual-Analytics-Machine-Learning-for-Productivity-Prediction.git
cd SleepSync-Visual-Analytics-Machine-Learning-for-Productivity-Prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
streamlit run app.py
```

---

## 🔮 Roadmap

- [ ] Real-world sleep tracker dataset integration (Fitbit / WHOOP)
- [ ] User authentication + personal sleep log tracking
- [ ] Wearable device API integration
- [ ] LSTM-based time-series productivity forecasting
- [ ] Mobile-responsive layout optimization
- [ ] Expanded dataset across more professions

---

## 📚 Learning Outcomes

This project demonstrates:
- End-to-end ML pipeline development (data → features → model → deployment)
- Domain-informed feature engineering for wellness data
- Interactive ML experimentation UI design
- Multi-model comparison and evaluation
- Streamlit dashboard development with custom theming

---

## 👨‍💻 Author

**Bandapalli Chirudeep**
MS Computer Science · UNC Charlotte · AI & Data Engineering

[![LinkedIn](https://img.shields.io/badge/LinkedIn-chirudeepbandapalli-blue?style=flat&logo=linkedin)](https://linkedin.com/in/chirudeepbandapalli)
[![GitHub](https://img.shields.io/badge/GitHub-Chirudeep2002-black?style=flat&logo=github)](https://github.com/Chirudeep2002)
[![Portfolio](https://img.shields.io/badge/Portfolio-chirudeep--portfolio.vercel.app-green?style=flat)](https://chirudeep-portfolio.vercel.app)
