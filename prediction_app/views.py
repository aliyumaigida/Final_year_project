
import os
import joblib
import pandas as pd
import tensorflow as tf
from django.shortcuts import render

# Paths & model loading
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'prediction_app', 'models')

scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
feature_names = joblib.load(os.path.join(MODEL_DIR, 'feature_names.pkl'))

# Load the deep learning model
# model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'breast_cancer_model_saved'))
model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'breast_cancer_model_saved.keras'))

def about_page(request):
    context = {'feature_names': feature_names}

    if request.method == 'POST':
        try:
            # === CSV Upload Path ===
            if request.FILES.get('csv_file'):
                df = pd.read_csv(request.FILES['csv_file'])

                # Get the start and end indices from form
                start_index = int(request.POST.get('start_index', 0))
                end_index = int(request.POST.get('end_index', start_index + 1))

                # Validate index range
                if start_index < 0 or end_index > len(df) or start_index >= end_index:
                    raise ValueError(f"Invalid row range: start={start_index}, end={end_index}. CSV has {len(df)} rows.")

                # Select the row slice
                df = df.iloc[start_index:end_index]

                # Check for missing columns
                missing = [f for f in feature_names if f not in df.columns]
                if missing:
                    raise ValueError(f"Missing columns: {missing}")

                # Reorder columns
                df = df[feature_names]

                # Preprocess the data (scaling only, no PCA)
                scaled = scaler.transform(df)

                # Check the shape of the scaled data
                print(scaled.shape)  # Ensure this prints (n_samples, 30)

                # Reshape the data to match the model's input shape (None, 30, 1)
                scaled = scaled.reshape(scaled.shape[0], scaled.shape[1], 1)

                # Make predictions
                predictions = model.predict(scaled).flatten()

                # Prepare results
                results = [
                    f"Patient_ID {i + start_index}: {'Cancerous (Malignant)' if pred > 0.4 else 'Non-cancerous (Benign)'}"
                    for i, pred in enumerate(predictions)
                ]
                context['results'] = results

            # # === Manual Form Input Path ===
            # else:
            #     try:
            #         input_data = [float(request.POST.get(f)) for f in feature_names]
            #     except (TypeError, ValueError) as e:
            #         raise ValueError("Please fill in all fields with valid numbers.") from e

            #     df = pd.DataFrame([input_data], columns=feature_names)

            #     # Preprocess the data (scaling only, no PCA)
            #     scaled = scaler.transform(df)

            #     # Check the shape of the scaled data
            #     print(scaled.shape)  # Ensure this prints (1, 30)

            #     # Reshape the data to match the model's input shape (None, 30, 1)
            #     scaled = scaled.reshape(scaled.shape[0], scaled.shape[1], 1)

            #     # Make prediction
            #     prediction = model.predict(scaled).flatten()[0]
            #     result = "Cancerous (Malignant)" if prediction > 0.4 else "Non-cancerous (Benign)"
            #     context['result'] = result

        except Exception as e:
            print(f"Error: {str(e)}")  # Print the error for debugging
            context['error'] = str(e)

    return render(request, 'prediction_app/about.html', context)

def predict_page(request):
    return render(request, 'prediction_app/index.html')












