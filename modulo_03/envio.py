# %% [markdown]
# # Apriori Algorithm Intro
# 
# This is a support code for medium article called **Apriori Algorithm Intro** and 'Uma Introdução ao Algoritmo Apriori' (pt-BR) [https://medium.com/@bernardo.costa/uma-introdu%C3%A7%C3%A3o-ao-algoritmo-apriori-60b11293aa5a]

# %%
"""
.. module:: Apriori
    :platform: Unix, Windows

.. note::
    Apriori class to handle Apriori execution

.. moduleauthor:: `Bernardo Costa <bernardoantunescosta@gmail.com>`
"""

import pandas as pd


# create sample dataset
columns = ['ID', 'Beer', 'Diaper', 'Gum', 'Soda', 'Snack']
dataset = [[1, 1, 1, 1, 1, 0],
           [2, 1, 1, 0, 0, 0],
           [3, 1, 1, 1, 0, 1],
           [4, 1, 1, 0, 1, 1],
           [5, 0, 1, 0, 1, 0],
           [6, 0, 1, 0, 0, 0],
           [7, 0, 1, 0, 0, 0],
           [8, 0, 0, 0, 1, 1],
           [9, 0, 0, 0, 1, 1]]


df = pd.DataFrame(dataset, columns=columns) 
df

# %%
from mlxtend.frequent_patterns import apriori

class Apriori:
    """Apriori Class. Its has Apriori steps."""
    threshold = 0.5
    df = None

    def __init__(self, df, threshold=None, transform_bol=False):
        """Apriori Constructor. 

        :param pandas.DataFrame df: transactions dataset (1 or 0).
        :param float threshold: set threshold for min_support.
        :return: Apriori instance.
        :rtype: Apriori
        """

        self._validate_df(df)

        self.df = df
        if threshold is not None:
            self.threshold = threshold

        if transform_bol:
            self._transform_bol()

    def _validate_df(self, df=None):
        """Validade if df exists. 

        :param pandas.DataFrame df: transactions dataset (1 or 0).
        :return: 
        :rtype: void
        """

        if df is None:
            raise Exception("df must be a valid pandas.DataDrame.")


    def _transform_bol(self):
        """Transform (1 or 0) dataset to (True or False). 

        :return: 
        :rtype: void
        """

        for column in self.df.columns:
            self.df[column] = self.df[column].apply(lambda x: True if x == 1 else False)


    def _apriori(self, use_colnames=False, max_len=None, count=True):
        """Call apriori mlxtend.frequent_patterns function. 

        :param bool use_colnames: Flag to use columns name in final DataFrame.
        :param int max_len: Maximum length of itemsets generated.
        :param bool count: Flag to count length of the itemsets.
        :return: apriori DataFrame.
        :rtype: pandas.DataFrame
        """
    
        apriori_df = apriori(
                    self.df, 
                    min_support=self.threshold,
                    use_colnames=use_colnames, 
                    max_len=max_len
                )
        if count:
            apriori_df['length'] = apriori_df['itemsets'].apply(lambda x: len(x))

        return apriori_df

    def run(self, use_colnames=False, max_len=None, count=True):
        """Apriori Runner Function.

        :param bool use_colnames: Flag to use columns name in final DataFrame.
        :param int max_len: Maximum length of itemsets generated.
        :param bool count: Flag to count length of the itemsets.
        :return: apriori DataFrame.
        :rtype: pandas.DataFrame
        """

        return self._apriori(
                        use_colnames=use_colnames,
                        max_len=max_len,
                        count=count
                    )

    def filter(self, apriori_df, length, threshold):
        """Filter Apriori DataFrame by length and  threshold.

        :param pandas.DataFrame apriori_df: Apriori DataFrame.
        :param int length: Length of itemsets required.
        :param float threshold: Minimum threshold nrequired.
        :return: apriori filtered DataFrame.
        :rtype:pandas.DataFrame
        """
        
        if 'length' not in apriori_df.columns:
            raise Exception("apriori_df has no length. Please run the Apriori with count=True.")

        return apriori_df[ (apriori_df['length'] == length) & (apriori_df['support'] >= threshold) ]


# %%
# Running Apriori 

if 'ID' in df.columns: del df['ID'] # ID is not relevant to apriori 

apriori_runner = Apriori(df, threshold=0.4, transform_bol=True)
apriori_df = apriori_runner.run(use_colnames=True)
apriori_df

# %%
# Showing only pairs with support granter than 0.41
apriori_runner.filter(apriori_df, length=2, threshold=0.41)

# %%
# Criando dataset disponibilizado no livro

columns = ['No.', 'Leite', 'Café', 'Cerveja', 'Pão', 'Manteiga', 'Arroz', 'Feijão']
dataset = [[1, 0, 1, 0, 1, 1, 0, 0],
           [2, 1, 0, 1, 1, 1, 0, 0],
           [3, 0, 1, 0, 1, 1, 0, 0],
           [4, 1, 1, 0, 1, 1, 0, 0],
           [5, 0, 0, 1, 0, 0, 0, 0],
           [6, 0, 0, 0, 0, 1, 0, 0],
           [7, 0, 0, 0, 1, 0, 0, 0],
           [8, 0, 0, 0, 0, 0, 0, 1],
           [9, 0, 0, 0, 0, 0, 1, 1],
           [10, 0, 0, 0, 0, 0, 1, 0],
           ]


df = pd.DataFrame(dataset, columns=columns) 
df



# %%
# Executando o Apriori

if 'ID' in df.columns: del df['ID'] # ID is not relevant to apriori 

apriori_runner = Apriori(df, threshold=0.3, transform_bol=True)
apriori_df = apriori_runner.run(use_colnames=True)
apriori_df.sort_values(by='length', ascending=False)

# %% [markdown]
# Com base na técnica Apriori, considerando o suporte de 0.3 como mínimo foram Pão e Manteiga, seguido de (Manteiga, Café) e (Pão, Café)
# Além disso, obtivemos as regras:
# - Quem compra Café, também compra Manteiga e Pão.
# - Quem compra Café, compra Pão.
# - Quem compra Manteiga, compra Pão.
# - Há chance de comprar Café, Pão e Manteiga isoladamente.

# %%



