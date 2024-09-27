# Importa as bibliotecas necessárias.
import tensorflow as tf
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Carrega o dataset Wine (ID 109) da UC Irvine Machine Learning Repository.
wine = fetch_ucirepo(id=109)

# As features (X) e os targets (y) são extraídos dos dados.
X = wine.data.features
y = wine.data.targets

# Exibe informações sobre o dataset (opcional).
print(wine.metadata)
print(wine.variables)

# Divide o dataset em conjunto de treinamento (67%) e teste (33%).
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Reindexe os rótulos para começar de 0 em vez de 1.(dava erro e tive ajuda do chatgpt pra corrigir)
y_train = y_train - 1
y_test = y_test - 1

# Cria um modelo sequencial no TensorFlow.
model = tf.keras.models.Sequential([
    # Camada de flattening.
    tf.keras.layers.Flatten(input_shape=(X.shape[1],)),  # Define o input_shape conforme o número de features do dataset Wine

    # Camada densa com 512 neurônios e função de ativação ReLU.
    tf.keras.layers.Dense(512, activation=tf.nn.relu),

    # Camada densa final com 3 neurônios (para 3 classes no dataset Wine) e função softmax.
    tf.keras.layers.Dense(3, activation=tf.nn.softmax)
])

# Compila o modelo.
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Treina o modelo com os dados de treinamento, armazenando o histórico.
history = model.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test))

# Avalia o modelo com os dados de teste.
model.evaluate(x_test, y_test)

# Faz previsões com os dados de teste.
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)  # Converte as probabilidades em classes previstas.

# Função para plotar gráficos de perda e acurácia.
def plot_history(history):
    # Plota a perda durante o treinamento e validação.
    plt.figure(figsize=(12, 4))

    # Plotando a perda
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Perda de treinamento')
    plt.plot(history.history['val_loss'], label='Perda de validação')
    plt.title('Perda durante o treinamento e validação')
    plt.xlabel('Época')
    plt.ylabel('Perda')
    plt.legend()

    # Plotando a acurácia
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Acurácia de treinamento')
    plt.plot(history.history['val_accuracy'], label='Acurácia de validação')
    plt.title('Acurácia durante o treinamento e validação')
    plt.xlabel('Época')
    plt.ylabel('Acurácia')
    plt.legend()

    # Exibe os gráficos
    plt.show()

# Chama a função para plotar o histórico de treinamento
plot_history(history)

# Calcula a matriz de confusão.
cm = confusion_matrix(y_test, y_pred_classes)

# Exibe a matriz de confusão.
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
plt.title('Matriz de Confusão')
plt.xlabel('Classe Prevista')
plt.ylabel('Classe Verdadeira')
plt.show()
