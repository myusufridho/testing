from flask import Flask, request, render_template
import joblib
import numpy as np

# Inisialisasi Flask app
app = Flask(__name__)

# Load model
random_forest_model = joblib.load('models/random_forest_model.pkl')
gbm_model = joblib.load('models/gbm_model.pkl')
svm_model = joblib.load('models/svm_model.pkl')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Ambil data dari form
    feature1 = float(request.form['feature1'])
    feature2 = float(request.form['feature2'])
    # Tambahkan fitur lainnya jika diperlukan

    features = np.array([feature1, feature2])  # Tambahkan semua fitur di sini

    # Lakukan inferensi dengan model
    rf_pred = random_forest_model.predict([features])[0]
    gbm_pred = gbm_model.predict([features])[0]
    svm_pred = svm_model.predict([features])[0]

    # Kembalikan hasil prediksi ke template HTML
    return render_template('index.html',
                           prediction=f'Random Forest: {rf_pred}, GBM: {gbm_pred}, SVM: {svm_pred}')

if __name__ == '__main__':
    app.run(debug=True)
