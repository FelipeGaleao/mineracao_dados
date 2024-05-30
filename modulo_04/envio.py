# %% [markdown]
# # Introdução
# Para esse módulo, irei utilizar a técnica do Naive Bayes que O Naive Bayes é um algoritmo de aprendizado de máquina baseado em probabilidade e no Teorema de Bayes. Ele assume independência entre os atributos, sendo usado para classificação com rapidez e simplicidade, especialmente em grandes conjuntos de dados. 

# %%
import pandas as pd
import numpy as np 
import sklearn

# %%
df = pd.read_csv('agaricus-lepiota.data')
df = df.rename(columns={
    'p': 'poisonous',
    'x': 'cap-shape',
    's': 'cap-surface',
    'n': 'cap-color',
    't': 'bruises',
    'p.1': 'odor',
    'f': 'gill-attachment',
    'c': 'gill-spacing',
    'n.1': 'gill-size',
    'k': 'gill-color',
    'e': 'stalk-shape',
    'e.1': 'stalk-root',
    's.1': 'stalk-surface-above-ring',
    's.2': 'stalk-surface-below-ring',
    'w': 'stalk-color-above-ring',
    'w.1': 'stalk-color-below-ring',
    'p.2': 'veil-type',
    'w.2': 'veil-color',
    'o': 'ring-number',
    'p.3': 'ring-type',
    'k.1': 'spore-print-color',
    's.3': 'population',
    'u': 'habitat'
})
# Exibindo completude da base
print(df.isnull().sum())
print(df.count())

# %%
# encondar os valores da base, para desencodarmos podemos usar o inverse_transform do SKLearn
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_encodado  = df.apply(le.fit_transform)
df_encodado.head()

# %%
# Separando a base em treino e teste
from sklearn.model_selection import train_test_split
X = df_encodado.drop('poisonous', axis=1)
y = df_encodado['poisonous']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Treinando o modelo Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Testando o modelo
y_pred = gnb.predict(X_test)

# Avaliando o modelo
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

# Matriz de confusão e o valor predito
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)

# Exibindo a matriz de confusão
pd.crosstab(y_test, y_pred, rownames=['Real'], colnames=['Predito'], margins=True)

# %%
# Mostrando a feature importance para cada um dos atributos
importances = gnb.theta_
importances = np.array(importances)
importances = importances[0]
importances = importances/sum(importances)
importances = np.array(importances)
importances = importances.reshape(1, 22)
importances = pd.DataFrame(importances, columns=X.columns)
importances = importances.T
importances = importances.rename(columns={0: 'importance'})
importances = importances.sort_values(by='importance', ascending=False)
print(importances)


# %% [markdown]
# # Resutaldo
# 
# Foram utilizadas 8123 registros, sendo utilizado 30% para testes e 70% para validação. Ao treinar o modelo Naive Bayes, foi encontrado a acurácia de 92%, onde no livro foi encontrado uma acurácia de 97%.
# 


