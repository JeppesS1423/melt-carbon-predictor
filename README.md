# AI-Powered Steel Melt Carbon Predictor 🏭

*A machine learning tool to optimize carbon content prediction in foundry environments*

![Streamlit App Demo](https://img.shields.io/badge/Demo-Online-green?style=flat&logo=streamlit) 
![Python](https://img.shields.io/badge/Python-3.12%2B-blue?logo=python)

## 🔍 Problem Statement
In steel foundries, **precise carbon control** is critical for:
- Meeting product specifications (±0.05% tolerance)
- Reducing rework/scrap costs (up to $500k/year savings)
- Minimizing energy waste from off-spec melts

Traditional methods rely on static calculations that ignore:
- Dynamic process conditions (temperature fluctuations)
- Scrap material interactions
- Furnace state degradation over time

## 🤖 AI Solution
This tool predicts final carbon content more accurately using:
```python
RandomForestRegressor(
    n_estimators=150,  # Ensemble of 150 decision trees
    max_depth=15,      # Captures complex non-linear relationships
)
```

Key Features:

* Processes real-world foundry data (scrap mixes, added carbon, melt times)

* Compares AI predictions vs. traditional calculations

* Identifies critical process parameters via feature importance

🚀 How It Works

* Input Process Parameters

    * Temperature (°C), melt time (s), scrap composition

    * Added carbon amount (kg)

* Model Predicts Carbon Content

    * Accounts for hidden process dynamics

    * Outputs deviation from expected value

* Optimization Insights

    * Recommends adjustments to hit target carbon


🛠️ Technical Setup
```bash

# Clone repository
git clone https://github.com/JeppesS1423/melt-carbon-predictor
cd melt-carbon-predictor

# Install dependencies
pip install -r requirements.txt  # streamlit pandas scikit-learn plotly

# Run the app
streamlit run main.py
```

🌐 Live Demo

https://melt-carbon-predictor-ddkwzitma27mkjtaxe8qxp.streamlit.app/

🧠 Key Learnings

* Process Matters: Melt time and temperature are as important as scrap composition

* Data Quality: Historical melt logs are gold for AI training

* Deployability: Streamlit bridges the gap between ML and shop-floor use