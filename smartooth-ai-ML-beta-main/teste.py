import joblib
import numpy as np

# Carregue o modelo e teste a previsão
try:
    model = joblib.load('models/model.pkl')
    # Dado de teste para verificar a previsão
    test_data = np.array([[30, 1, 2]])
    test_prediction = model.predict(test_data)
    print("Model loaded and prediction successful:", test_prediction)
except Exception as e:
    print(f"Error loading model or predicting: {str(e)}")