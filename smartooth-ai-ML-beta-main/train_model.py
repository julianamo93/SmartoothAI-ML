from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np
import os

# Verificar se o diretório 'models' existe, caso contrário, criá-lo
if not os.path.exists('models'):
    os.makedirs('models')

# Exemplo de dados fictícios para treino (substitua pelos dados reais)
X_train = np.array([[30, 1, 2], [25, 0, 1], [40, 1, 3]])
y_train = np.array([1, 0, 1])  # Rótulos: 1 para necessidade de tratamento, 0 para acompanhamento regular

# Treinando o modelo
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Salvando o modelo treinado
joblib.dump(model, 'models/model.pkl')
print(f"Modelo treinado e salvo em ./Models")