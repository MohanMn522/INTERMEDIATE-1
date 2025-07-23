Credit Card Fraud Detection
Overview
This project uses machine learning to detect and prevent fraudulent credit card transactions in real-time. It includes:

Data preprocessing & feature scaling

Model training with class imbalance handling

Real-time fraud detection API (FastAPI)

Monitoring dashboard (Streamlit)

Compliance & continuous monitoring best practices

📁 Project Structure
bash
Copy
Edit
credit-card-fraud-detection/
│
├── creditcard.csv             # Dataset (Kaggle)
├── train_model.py             # Preprocess and train model
├── scaler.pkl                 # Saved scaler object
├── model.pkl                  # Trained model
│
├── api/
│   └── app.py                 # FastAPI app for real-time prediction
│
├── dashboard/
│   └── app.py                 # Streamlit dashboard
│
├── requirements.txt           # Python dependencies
└── README.md                  # This file
🚀 Getting Started
1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection
2. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
3. Get the Dataset
Download creditcard.csv from the Kaggle dataset and place it in the project root.

4. Train the Model
bash
Copy
Edit
python train_model.py
This will generate:

model.pkl: trained fraud detection model

scaler.pkl: preprocessing scaler

🧠 Model Details
Algorithm: Random Forest Classifier

Preprocessing:

Feature scaling with StandardScaler

Imbalance handling using RandomOverSampler

Evaluation: Classification report and ROC-AUC

Target: Class (1 = Fraud, 0 = Legit)

🔌 Real-Time API
Start FastAPI Server
bash
Copy
Edit
uvicorn api.app:app --reload
Endpoint
POST /predict/: Accepts transaction data and returns prediction

Example request:

json
Copy
Edit
{
  "Time": 12345,
  "V1": -1.3598,
  "V2": 1.1918,
  ...
  "Amount": 149.62
}
📊 Monitoring Dashboard
Start Streamlit App
bash
Copy
Edit
streamlit run dashboard/app.py
Use the dashboard to:

Simulate transactions

Monitor fraud detection responses

Get alerts if fraud is detected

🛡️ Compliance & Monitoring
Data anonymized (no personal info)

Model explainability via tree-based classifiers

Real-time logs for monitoring predictions

Ready for SOC2/GDPR alignment with proper auditing

👥 Collaboration
This project encourages collaboration among:

Data scientists (modeling & analysis)

Engineers (deployment & APIs)

Compliance teams (regulations & data handling)

📈 Future Improvements
Deploy to cloud (AWS/GCP)

Use streaming data (Kafka, Spark)

Add drift detection & alerts

Support model retraining pipelines (MLflow, Airflow)

📚 References
Kaggle Credit Card Fraud Dataset

Scikit-learn

FastAPI

Streamlit

Imbalanced-learn
