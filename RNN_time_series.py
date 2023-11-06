import pandas as pd

#dataset extracted from: https://br.financas.yahoo.com/
base = pd.read_csv('petr4.csv')
base = base.dropna()

base = base.iloc[:,1].values
import matplotlib.pyplot as plt
#%matplotlib inline
plt.plot(base)

#gets 30 values
periodos = 30 
#predicts 1
previsao_futura = 1 # horizonte

X = base[0:(len(base) - (len(base) % periodos))]
# -1 equals None
#batch trainning
X_batches = X.reshape(-1, periodos, 1)

y = base[1:(len(base) - (len(base) % periodos)) + previsao_futura]
y_batches = y.reshape(-1, periodos, 1)

#test atts
X_teste = base[-(periodos + previsao_futura):]
X_teste = X_teste[:periodos]
X_teste = X_teste.reshape(-1, periodos, 1)
#test labels
y_teste = base[-(periodos):]
y_teste = y_teste.reshape(-1, periodos, 1)

import tensorflow as tf
#reset form memory
tf.reset_default_graph()

#1 
entradas = 1
#it is neccessary to test many values to hidden layer
neuronios_oculta = 100
#return just a value (regression)
neuronios_saida = 1

#placeholders
xph = tf.placeholder(tf.float32, [None, periodos, entradas])
yph = tf.placeholder(tf.float32, [None, periodos, neuronios_saida])

#celula = tf.contrib.rnn.BasicRNNCell(num_units = neuronios_oculta, activation = tf.nn.relu)
#celula = tf.contrib.rnn.LSTMCell(num_units = neuronios_oculta, activation = tf.nn.relu)
# camada saída
#celula = tf.contrib.rnn.OutputProjectionWrapper(celula, output_size = 1)

def cria_uma_celula():
    return tf.contrib.rnn.LSTMCell(num_units = neuronios_oculta, activation = tf.nn.relu)

def cria_varias_celulas():
    celulas =  tf.nn.rnn_cell.MultiRNNCell([cria_uma_celula() for i in range(4)])
    return tf.contrib.rnn.DropoutWrapper(celulas, output_keep_prob = 0.1)

celula = cria_varias_celulas()
# camada saída
celula = tf.contrib.rnn.OutputProjectionWrapper(celula, output_size = 1)



saida_rnn, _ = tf.nn.dynamic_rnn(celula, xph, dtype = tf.float32)
#mae
erro = tf.losses.mean_squared_error(labels = yph, predictions = saida_rnn)
#adam optimizer
otimizador = tf.train.AdamOptimizer(learning_rate = 0.001)

treinamento = otimizador.minimize(erro)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoca in range(1000):
        _, custo = sess.run([treinamento, erro], feed_dict = {xph: X_batches, yph: y_batches})
        if epoca % 100 == 0:
            print(epoca + 1, ' erro: ', custo)
    
    #predicted values
    previsoes = sess.run(saida_rnn, feed_dict = {xph: X_teste})
    
import numpy as np
y_teste.shape

#reorganizing the structure of data with ravel
y_teste2 = np.ravel(y_teste)

#doing the same thing as above
previsoes2 = np.ravel(previsoes)

from sklearn.metrics import mean_absolute_error
#calculate the MAE between predicted values and correct values
mae = mean_absolute_error(y_teste2, previsoes2)

#Plotting results 
plt.plot(y_teste2, '*', markersize = 10, label = 'Valor real')
plt.plot(previsoes2, 'o', label = 'Previsões')
plt.legend()

#Plotting results 2
plt.plot(y_teste2, label = 'Valor real')
plt.plot(y_teste2, 'w*', markersize = 10, color = 'red')
plt.plot(previsoes2, label = 'Previsões')
plt.legend()
























