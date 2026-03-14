import numpy as np
import pandas as pd
import pickle
import io
import os
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from flask import Flask, request, render_template
from keras.models import load_model

app = Flask(__name__)

model = load_model('theft.h5')
sc = pickle.load(open('StandardScaler.pk', 'rb'))

FEATURE_COLS = ['energy_median', 'energy_mean', 'energy_max',
                'energy_count', 'energy_std', 'energy_sum', 'energy_min']

UPLOADED_DATA_PATH = 'uploaded_dataset.csv'

def get_active_dataset():
    """Returns uploaded dataset if exists, otherwise falls back to test data."""
    if os.path.exists(UPLOADED_DATA_PATH):
        return pd.read_csv(UPLOADED_DATA_PATH)
    return pd.read_csv('test_data/test_data.csv')


@app.route("/")
@app.route("/index")
def index():
    return render_template('index.html')


@app.route('/login')
def login():
    return render_template('login.html')


@app.route('/upload')
def upload():
    return render_template('upload.html')


@app.route('/preview', methods=["POST"])
def preview():
    if request.method == 'POST':
        dataset = request.files['datasetfile']
        df = pd.read_csv(dataset, encoding='unicode_escape')
        df.to_csv(UPLOADED_DATA_PATH, index=False)
        df.set_index('Id', inplace=True)
        return render_template("preview.html", df_view=df)


@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    return render_template('prediction.html')


@app.route('/predict', methods=['POST'])
def predict():
    int_feature = [x for x in request.form.values()]
    prediction = model.predict(sc.transform(np.array([int_feature])))
    if prediction > 0.5:
        result = "Unfaithful (Theft Detected)"
    else:
        result = "Faithful (No Theft)"
    return render_template('prediction.html', prediction_text=result)


@app.route('/chart')
def chart():
    df = get_active_dataset()

    # If no label column, generate predictions from model to use as labels
    if 'label' not in df.columns:
        X_scaled    = sc.transform(df[FEATURE_COLS].values)
        df['label'] = (model.predict(X_scaled) > 0.5).astype(int).flatten()

    total            = len(df)
    faithful_count   = int((df['label'] == 0).sum())
    unfaithful_count = int((df['label'] == 1).sum())

    return render_template('chart.html',
                           faithful=faithful_count,
                           unfaithful=unfaithful_count,
                           total=total)


@app.route('/performance')
def performance():
    df         = get_active_dataset()
    has_labels = 'label' in df.columns

    X           = df[FEATURE_COLS].values
    X_scaled    = sc.transform(X)
    y_pred_prob = model.predict(X_scaled)
    y_pred      = (y_pred_prob > 0.5).astype(int).flatten()

    total      = len(y_pred)
    img_base64 = None
    accuracy   = None
    correct    = None

    if has_labels:
        y_true = df['label'].values

        # Generate confusion matrix only when ground truth labels exist
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=['Faithful (0)', 'Unfaithful (1)'])
        disp.plot(ax=ax, colorbar=True, cmap='Blues')
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold', pad=12)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        accuracy = round((y_pred == y_true).mean() * 100, 2)
        correct  = int((y_pred == y_true).sum())

    return render_template('performance.html',
                           confusion_matrix_img=img_base64,
                           has_labels=has_labels,
                           accuracy=accuracy,
                           total=total,
                           correct=correct)


if __name__ == "__main__":
    app.run(debug=True)
