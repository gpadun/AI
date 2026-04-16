import pandas as pd

class Question:
    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        # Pega o valor da coluna específica na linha que estamos testando
        val = example[self.column]
        if val >= self.value:
            return True
        else:
            return False

class Leaf:
    def __init__(self, df):
        self.predictions = df['species'].value_counts().to_dict()

class Decision_node:
    def __init__(self, question, true_branch, false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch

class DecisionTree:
    def __init__(self):
       self.root = None

    def _gini(self, df):
        impurity = 1
        prob_sum = 0
        data_lableds = df['species'].value_counts().to_list()
        probabilities = []
        for value in data_lableds:
            probabilities.append(value/float(df.shape[0]))
        for probability in probabilities:
            probability = probability**2
            prob_sum += probability
        impurity -= prob_sum
        return impurity


    def _information_gain(self, left_node, right_node, current_impurity):
        if len(left_node) + len(right_node) == 0:
            return 0
        weight_left = float(len(left_node)) / (float(len(left_node)) + float(len(right_node)))
        weight_right = 1 - weight_left
        return current_impurity - (weight_left*self._gini(left_node) + weight_right*self._gini(right_node))


    def _partition(self, df, question):
        true_rows = df[df[question.column]>= question.value]
        false_rows = df[df[question.column]< question.value]
        return true_rows, false_rows


    def _best_split(self, df):
        best_gain = 0
        best_question = None
        current_impurity = self._gini(df)
        features = df.columns.drop('species')
        for column in features:
            single_values = df[column].unique()
            for val in single_values:
                question = Question(column, val)
                split = self._partition(df, question)
                info_gain = self._information_gain(split[0], split[1], current_impurity)
                if info_gain > best_gain:
                    best_gain = info_gain
                    best_question = question
        return best_gain, best_question


    def _build_tree(self, df):
        gain, question = self._best_split(df)
        if gain == 0:
            return Leaf(df)
        true_rows, false_rows = self._partition(df, question)
        true_branch = self._build_tree(true_rows)
        false_branch = self._build_tree(false_rows)
        node = Decision_node(question, true_branch, false_branch)
        return node

    def predict(self, df_teste):
        """Recebe um DataFrame de teste e devolve uma lista com as previsões."""
        previsoes = []
        # Iteramos sobre cada linha do DataFrame de teste
        for _, row in df_teste.iterrows():
            # Pegamos o dicionário da folha
            resultado_dict = self._classify(row, self.root)
            # Descobrimos qual é a chave (espécie) com o maior número de votos
            classe_vencedora = max(resultado_dict, key=resultado_dict.get)
            previsoes.append(classe_vencedora)

        return previsoes

    def _classify(self, row, node):
        """Navega recursivamente pela árvore até achar uma folha."""
        # Se chegamos em uma folha, devolvemos o dicionário de previsões
        if isinstance(node, Leaf):
            return node.predictions
        # Se não é folha, usamos a pergunta para saber se vamos para a esquerda ou direita
        if node.question.match(row):
            return self._classify(row, node.true_branch)
        else:
            return self._classify(row, node.false_branch)

    def fit(self, df):
       self.root = self._build_tree(df)
