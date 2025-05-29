# app.py
from flask import Flask, render_template, request, redirect, url_for, flash
import os
import numpy as np
from werkzeug.utils import secure_filename
import time
import sys

# Add model directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(current_dir, 'model')
sys.path.append(model_dir)

app = Flask(__name__)
app.secret_key = 'leaf_disease_detection_secret_key'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    # Check if model is trained
    model_path = os.path.join(model_dir, 'model.pth')
    labels_path = os.path.join(model_dir, 'class_labels.json')

    if not os.path.exists(model_path) or not os.path.exists(labels_path):
        flash("Warning: Model not trained yet. Please train the model before using the application.")

    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    model_path = os.path.join(model_dir, 'model.pth')
    labels_path = os.path.join(model_dir, 'class_labels.json')

    if not os.path.exists(model_path) or not os.path.exists(labels_path):
        flash("Error: Model not trained yet. Please train the model before using the application.")
        return redirect(url_for('index'))

    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp = str(int(time.time()))
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            from model.model import predict_disease
            from utils.treatments import get_treatment

            disease_name, confidence = predict_disease(filepath)
            disease_name = disease_name.strip().replace('\n', '').replace('\r', '')  # Clean label
            print("Predicted Disease:", repr(disease_name))  # Debugging output

            treatment = get_treatment(disease_name)

            return render_template('result.html', 
                                   image_path=f"/static/uploads/{filename}",
                                   disease=disease_name,
                                   confidence=confidence,
                                   treatment=treatment)
        except Exception as e:
            flash(f"Error during prediction: {str(e)}")
            return redirect(url_for('index'))

    flash('File type not allowed')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
