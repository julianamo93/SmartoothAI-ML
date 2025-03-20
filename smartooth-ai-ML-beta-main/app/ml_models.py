import random
import joblib
import numpy as np

# Carregar o modelo treinado
model = joblib.load('models/model.pkl')

def get_dental_tips(patient_id):
    # Lógica de exemplo para retornar dicas odontológicas
    tips = [
        "Escove seus dentes duas vezes ao dia.",
        "Use fio dental regularmente.",
        "Consuma menos açúcar para evitar cáries."
    ]
    return random.sample(tips, 2)  # Retorna 2 dicas aleatórias

def get_recommendations(patient_id):
    # Lógica de exemplo para retornar recomendações odontológicas
    recommendations = [
        "Agende uma consulta de limpeza.",
        "Faça um check-up para verificar o estado das suas gengivas.",
        "Tente usar um creme dental com flúor."
    ]
    return random.sample(recommendations, 2)  # Retorna 2 recomendações aleatórias

def predict(input_data):
    """
    Função para realizar a previsão com o modelo treinado.
    O modelo espera um dicionário com 'age', 'history' e 'severity' como entradas.
    """
    features = np.array([input_data['age'], input_data['history'], input_data['severity']]).reshape(1, -1)
    prediction = model.predict(features)
    return prediction[0]  # Retorna um único valor de previsão

def generate_recommendation(prediction):
    """
    Gera uma recomendação baseada na previsão do modelo.
    Se a previsão for 1, recomenda tratamento preventivo.
    Caso contrário, sugere acompanhamento regular com o dentista.
    """
    return "Recomendamos um tratamento preventivo." if prediction == 1 else "Mantenha acompanhamento regular com o dentista."
