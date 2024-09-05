import numpy as np

# Função de ativação sigmoide
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivada da função sigmoide
def sigmoid_derivative(x):
    return x * (1 - x)

# Dados de treinamento
X = np.array([[1, 1, 0, 1, 0],
              [1, 0, 1, 0, 1],
              [1, 0, 0, 1, 0],
              [1, 0, 0, 0, 1]])

yd = np.array([[1, 0],
               [0, 1],
               [1, 0],
               [0, 1]])

# Inicialização das sinapses
np.random.seed(1)
w1 = 2 * np.random.random((5, 4)) - 1
w2 = 2 * np.random.random((4, 2)) - 1

# Taxa de aprendizado
learning_rate = 0.1

# Treinamento da rede neural
for training in range(10000):
    # Forward propagation
    layer1 = sigmoid(np.dot(X, w1))
    layer2 = sigmoid(np.dot(layer1, w2))

    # Backpropagation
    layer2_error = yd - layer2  # Calcula o erro da saída
    layer2_delta = layer2_error * sigmoid_derivative(layer2)  # Corrigir aqui para usar a derivada corretamente

    layer1_error = layer2_delta.dot(w2.T)
    layer1_delta = layer1_error * sigmoid_derivative(layer1)

    # Atualização dos pesos
    w2 += layer1.T.dot(layer2_delta) * learning_rate
    w1 += X.T.dot(layer1_delta) * learning_rate

# Função para prever a doença
def prever_doenca(sintomas):
    layer1 = sigmoid(np.dot(sintomas, w1))
    layer2 = sigmoid(np.dot(layer1, w2))
    # Aqui, layer2 será um vetor com duas saídas, então podemos indexar sem erro
    if layer2[0] > layer2[1]:
        return "Gripe"
    else:
        return "Dengue"

# Teste com novos exemplos
novos_exemplos = np.array([[1, 1, 0, 0, 0],
                           [0, 0, 1, 1, 0],
                           [1, 0, 0, 1, 0],
                           [0, 1, 1, 0, 0]])

print("Previsão de doenças para novos exemplos:")
for exemplo in novos_exemplos:
    print(f"Sintomas: {exemplo} -> Doença prevista: {prever_doenca(exemplo)}")
