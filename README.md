# 🛡️ AI Cyber Threat Detection

### Machine Learning Based Network Intrusion Detection System

An AI-powered cybersecurity system that analyzes network traffic and detects potentially malicious activity using **Machine Learning**.

The project uses a **Random Forest classifier** trained on the **UNSW-NB15 dataset** to classify network traffic as **Normal** or **Attack**, estimate attack probability, assign risk levels, and provide an interactive Streamlit dashboard for threat analysis.

---

## 🚀 Project Overview

Modern networks generate a huge amount of traffic, making manual monitoring difficult.

This project aims to automate the first stage of network threat detection by analyzing network traffic features and identifying suspicious patterns using Machine Learning.

The system provides:

* Network traffic preprocessing
* Categorical feature encoding
* Numerical feature processing
* Random Forest model training
* Binary attack detection
* Attack probability estimation
* Risk-level classification
* Confusion matrix analysis
* Threat log filtering
* Individual network-record inspection
* Interactive Streamlit dashboard

> **Note:** The current version performs analysis on the UNSW-NB15 test dataset. It is a machine-learning demonstration and is not a live packet-capture or production intrusion prevention system.

---

## 🧠 System Architecture

```text
                 UNSW-NB15 Dataset
                         │
                         ▼
                Data Preprocessing
                         │
             ┌───────────┴───────────┐
             │                       │
       Numerical Features     Categorical Features
             │                       │
             └───────────┬───────────┘
                         │
                         ▼
                  Feature Encoding
                         │
                         ▼
                Random Forest Model
                         │
                         ▼
                 Threat Prediction
                         │
              ┌──────────┴──────────┐
              │                     │
          Normal                 Attack
              │                     │
              └──────────┬──────────┘
                         │
                         ▼
                Attack Probability
                         │
                         ▼
                    Risk Level
                         │
                         ▼
               Threat Analysis CSV
                         │
                         ▼
               Streamlit Dashboard
```

---

## 📊 Dataset

This project uses the **UNSW-NB15** network intrusion detection dataset.

The dataset contains network traffic records with features describing network connections and attack behavior.

### Dataset Size

| Dataset  | Records | Features |
| -------- | ------: | -------: |
| Training | 175,341 |       45 |
| Testing  |  82,332 |       45 |

### Attack Categories

The dataset contains categories including:

* Normal
* Generic
* Exploits
* Fuzzers
* DoS
* Reconnaissance
* Analysis
* Backdoor
* Shellcode
* Worms

---

## 🔧 Data Preprocessing

The preprocessing pipeline performs:

1. Dataset loading
2. Missing-value analysis
3. Duplicate detection
4. Target separation
5. Numerical feature identification
6. Categorical feature identification
7. Feature encoding
8. Transformation of training and testing data
9. Saving the processed datasets
10. Saving the preprocessing pipeline

The trained preprocessing pipeline is stored as:

```text
models/preprocessor.pkl
```

---

## 🤖 Machine Learning Model

### Random Forest Classifier

The primary model used in this project is a **Random Forest classifier**.

Random Forest was selected because it performs well on structured/tabular data and can capture nonlinear relationships between network traffic features.

The model predicts:

```text
0 → Normal
1 → Attack
```

It also provides attack probabilities used by the risk-classification system.

The trained model is stored as:

```text
models/random_forest.pkl
```

Git LFS is used for the large model file.

---

## 📈 Model Performance

The model was evaluated on the UNSW-NB15 test dataset.

| Metric        |     Result |
| ------------- | ---------: |
| Accuracy      | **89.24%** |
| Precision     | **85.08%** |
| Attack Recall | **97.56%** |
| F1 Score      | **90.90%** |
| ROC-AUC       | **98.09%** |

### Confusion Matrix

|               | Predicted Normal | Predicted Attack |
| ------------- | ---------------: | ---------------: |
| Actual Normal |           29,247 |            7,753 |
| Actual Attack |            1,107 |           44,225 |

### Interpretation

The model detected **97.56% of attacks**, meaning the missed-attack rate was approximately **2.44%** on the test dataset.

The relatively high false-positive rate indicates that some normal network traffic is classified as suspicious. This is an important area for future optimization.

---

## 🚨 Risk Classification

The system converts attack probability into four risk levels:

```text
                Attack Probability
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
      LOW              MEDIUM            HIGH
                         │
                         ▼
                      CRITICAL
```

Current risk distribution from the test-data analysis:

| Risk Level | Records |
| ---------- | ------: |
| CRITICAL   |  39,403 |
| HIGH       |   6,611 |
| MEDIUM     |   6,106 |
| LOW        |  30,212 |

---

## 🖥️ Streamlit Dashboard

The project includes an interactive dashboard for analyzing the model's predictions.

### Dashboard Features

* 📊 Traffic statistics
* 🚨 Threat rate
* 🎯 Model performance
* 📈 Attack probability visualization
* 🔴 Risk-level distribution
* 🔍 Network threat logs
* 🔎 Individual traffic inspection
* 📋 Network traffic details
* 🧮 Confusion-matrix analysis

### Dashboard Preview

Add your screenshot here:

```text
screenshots/dashboard.png
```

Example Markdown:

```markdown
![AI Cyber Threat Detection Dashboard](screenshots/dashboard.png)
```

---

## 📁 Project Structure


AI-Cyber-Threat-Detection/
│
├── models/
│   ├── preprocessor.pkl
│   └── random_forest.pkl
│
├── processed_data/
│   └── threat_analysis_results.csv
│
├── src/
│   ├── app.py
│   ├── predict.py
│   ├── preprocessing.py
│   ├── train_model.py
│   └── threat_analysis.py
│
├── screenshots/
│   └── dashboard.png
│
├── .gitattributes
├── .gitignore
├── requirements.txt
└── README.md


## ⚙️ Technologies Used

### Programming

* Python

### Data Science

* Pandas
* NumPy
* Scikit-learn

### Machine Learning

* Random Forest
* Feature preprocessing
* Classification
* Model evaluation

### Visualization / Dashboard

* Streamlit
* PyDeck / Streamlit visualization components

### Model Management

* Joblib
* Git
* Git LFS

---



## 🔐 Security Scope

This project focuses on **defensive cybersecurity** and network intrusion detection.

It does not perform:

* Unauthorized network access
* Exploitation of systems
* Credential theft
* Malware deployment
* Offensive penetration testing

The goal is to identify suspicious network behavior and support defensive security analysis.

---

## 🚀 Future Improvements

Planned improvements include:

* [ ] Real-time network packet ingestion
* [ ] Live network monitoring
* [ ] Deep Learning based detection
* [ ] XGBoost / LightGBM model comparison
* [ ] Explainable AI using SHAP
* [ ] Attack-category classification
* [ ] Real-time alert notifications
* [ ] Network traffic visualization
* [ ] REST API for predictions
* [ ] Docker deployment
* [ ] Cloud deployment
* [ ] Improved false-positive reduction
* [ ] Model retraining pipeline
* [ ] MLOps monitoring

---

## 🎯 Learning Outcomes

Through this project, I worked with:

* Real-world cybersecurity data
* Data preprocessing
* Feature engineering
* Categorical encoding
* Machine Learning classification
* Random Forest
* Model evaluation
* Probability-based risk assessment
* Streamlit application development
* Model serialization
* Git and Git LFS
* Machine Learning deployment concepts

---

## 👨‍💻 Author

**Suman Kumar**

Python Developer | AI/ML Enthusiast

---

## ⭐ Project

If you find this project useful or interesting, consider giving the repository a ⭐.

**GitHub Repository:** 
https://github.com/sumankr62800-del/AI-Cyber-Threat-Detection
