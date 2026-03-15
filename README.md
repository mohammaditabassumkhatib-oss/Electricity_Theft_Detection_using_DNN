# Electricity Theft Detection in Smart Grids using Deep Neural Network

A machine learning web application that detects fraudulent electricity usage patterns in smart grids using a Deep Neural Network (DNN). Built as a Final Year Bachelor's Project.

---

## Problem Statement

Electricity theft is a major global issue causing billions of dollars in losses annually for utility companies. Traditional rule-based detection methods are slow and inaccurate. This project uses a **Deep Neural Network trained on smart meter energy consumption data** to automatically classify consumers as faithful or unfaithful — achieving **99% accuracy**.

---

## Model Architecture

| Component | Detail |
|---|---|
| Algorithm | Artificial Neural Network (ANN / DNN) |
| Clustering | Agglomerative Clustering (k=3) to label theft |
| Scaler | StandardScaler (saved as `StandardScaler.pk`) |
| Training Accuracy | 99% |
| Validation Accuracy | 99% |
| Output | Binary classification — Faithful (0) / Unfaithful (1) |

**Input features:**
- `energy_median`, `energy_mean`, `energy_max`
- `energy_count`, `energy_std`, `energy_sum`, `energy_min`

---

## Web Application

Built with **Flask** — upload a smart meter dataset and the system will:

- Preview the uploaded dataset
- Predict theft for individual consumers
- Show faithful vs unfaithful distribution chart
- Display confusion matrix and performance metrics live from the model

### Screenshots

#### Home Page
![Home Page](static/img/screenshots/home_page.png)
#### Prediction Input
![Prediction Input](static/img/screenshots/prediction_input.png)
#### Prediction Result
![Prediction Output](static/img/screenshots/prediction_output.png)
#### Chart
![Chart](static/img/screenshots/chart.png)
#### Performance Analysis
![Performance](static/img/screenshots/performance.png)
---

## How to Run Locally

### Prerequisites
- Python 3.8+
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/mohammaditabassumkhatib-oss/Electricity_Theft_Detection_using_DNN.git
cd electricity-theft-detection

# 2. Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
python app.py
```

Then open your browser and go to: **http://127.0.0.1:5000**

---

## Project Structure

```
electricity-theft-detection/
│
├── notebooks/
│   └── model.ipynb           # DNN training notebook
│
├── templates/                # HTML pages (Flask)
│   ├── index.html
│   ├── upload.html
│   ├── preview.html
│   ├── prediction.html
│   ├── chart.html
│   ├── performance.html
│   └── login.html
│
├── static/
│   ├── css/                  # Custom stylesheets
│   ├── js/                   # Custom JavaScript
│   └── img/                  # Images used in web
│
├── data/
│   └── test_data.csv         # Sample labelled dataset
│
├── paper/
│   └── Electricity_Theft_Detection_Paper.pdf  # Research paper
│
├── app.py                    # Flask backend
├── theft.h5                  # Trained DNN model
├── StandardScaler.pk         # Fitted scaler
├── requirements.txt
└── .gitignore
```

---

## Performance Results

| Metric | Class 0 (Faithful) | Class 1 (Unfaithful) |
|---|---|---|
| Precision | 1.00 | 0.97 |
| Recall | 0.99 | 1.00 |
| F1 Score | 1.00 | 0.95 |

**Confusion Matrix** is generated live from the model on the Performance Analysis page.

---

## Research Paper

A research paper accompanies this project covering the full methodology, dataset analysis, model design, and results. See [`paper/`](./paper/) for the PDF.

---

## Tech Stack

- **Backend:** Python, Flask
- **ML/DL:** TensorFlow, Keras, scikit-learn
- **Data:** Pandas, NumPy
- **Visualisation:** Matplotlib, Google Charts
- **Frontend:** HTML, CSS, Bootstrap 5

---

## Notes:

- If your dataset does not contain a `label` column, the app will run predictions using the trained model and display the distribution. The confusion matrix requires a labelled dataset.
- The trained model (`theft.h5`) and scaler (`StandardScaler.pk`) are included in the repo and loaded automatically on startup.
