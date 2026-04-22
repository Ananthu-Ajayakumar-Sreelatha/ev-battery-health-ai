# ⚡ EV Battery Health & Range Intelligence System

An end-to-end data engineering and AI pipeline that monitors 
lithium-ion battery health, predicts State of Health (SOH), 
detects thermal anomalies, and identifies abnormal discharge 
behaviour using real NASA battery data.

---

## 🎯 Business Problem
EV battery degradation is the biggest challenge facing 
electric vehicle manufacturers. Early detection of failing 
batteries reduces warranty costs, prevents safety incidents, 
and improves customer experience. This system predicts battery 
health **before** failure occurs.

---

## 🤖 AI Techniques Used

| Technique | Purpose | Result |
|-----------|---------|--------|
| **Random Forest Regressor** | Predicts State of Health (SOH) | 97.7% R², 0.98% MAE |
| **DBSCAN Clustering** | Identifies behaviour patterns | 15 clusters, 92 outliers |
| **Thermal Anomaly Detection** | Flags safety-critical events | 206 anomalies (7.5%) |
| **Capacity Integration** | Calculates real Ah capacity | Via trapezoidal method |

---

## 📊 Key Results
✅ 2,755 discharge cycles processed
✅ 34 batteries monitored
✅ 97.7% R² Score — Random Forest SOH prediction
✅ 0.98% Mean Absolute Error — less than 1% prediction error
✅ 15 behaviour clusters identified via DBSCAN
✅ 92 outlier cycles flagged (3.3%)
✅ 206 thermal anomaly events detected (7.5%)
✅ avg_temp identified as #1 degradation driver (55.97% importance)
---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python + Pandas | Data pipeline |
| scikit-learn | Random Forest + DBSCAN |
| NumPy | Capacity integration |
| Streamlit | Interactive dashboard |
| Plotly | Visualizations |

---

## 📊 Dashboard Features
- Per-battery SOH degradation curve
- Actual vs AI-predicted SOH comparison
- Capacity fade tracking over cycles
- Temperature profile with danger threshold
- Voltage behaviour monitoring
- DBSCAN behaviour cluster distribution
- Full 34-battery fleet health overview
- Automated Healthy/Degrading/Critical classification

---

## 📁 Dataset
**NASA Battery Dataset**
- 34 lithium-ion batteries
- 2,755 discharge cycles
- 6 sensor measurements per reading
- Voltage, Current, Temperature, Time

---

## 🚀 How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run data loading
python 01_load_data.py

# Run health pipeline
python 02_pipeline.py

# Run AI models
python 03_ai_models.py

# Launch dashboard
streamlit run 04_dashboard.py
```

---

## 📂 Project Structure
ev_battery_health/
├── 01_load_data.py       # Data loading & exploration
├── 02_pipeline.py        # Battery health pipeline & SOH
├── 03_ai_models.py       # Random Forest + DBSCAN + Thermal
├── 04_dashboard.py       # Streamlit dashboard
├── requirements.txt      # Dependencies
└── README.md             # Project documentation
---

## 👤 Author
MSc Data Science
University of Europe for Applied Sciences, Potsdam, Germany