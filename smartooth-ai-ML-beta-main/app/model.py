import joblib
import os
import numpy as np

# Função para carregar o modelo
def load_model():
    model_path = 'models/model.pkl'
    
    # Verificar se o arquivo existe
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo não encontrado no caminho {model_path}")
    
    # Carregar o modelo
    model = joblib.load(model_path)
    return model

# Função para fazer a previsão
def predict(input_data):
    model = load_model()
    features = np.array([input_data['age'], input_data['history'], input_data['severity']]).reshape(1, -1)
    prediction = model.predict(features)
    prediction_int = int(prediction[0])  # Converte para int
    print("Prediction value (should be int):", prediction_int)  # Verifica o tipo e valor
    return prediction_int


# Função para fornecer uma recomendação com base na previsão
def generate_recommendation(prediction):
    # Baseado na previsão, fornecemos uma recomendação
    if prediction == 1:
        return "Recomendamos um tratamento preventivo."
    else:
        return "Mantenha acompanhamento regular com o dentista."

