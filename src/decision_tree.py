import pandas as pd

class Question:
    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example): 
        val = example[self.column]
        if val >= self.value:
            return True
        else:
            return False

class LeafNode:
    def __init__(self, df, target_col):
        self.predictions = df[target_col].value_counts().to_dict()

class DecisionNode:
    def __init__(self, question, true_branch, false_branch): 
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch

class DecisionTree:
    def __init__(self, target_col):
       self.root = None
       self.target_col = target_col

    def _gini(self, df):
        impurity = 1
        prob_sum = 0
        data_labels = df[self.target_col].value_counts().to_list()
        probabilities = []
        for value in data_labels:
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


    def _best_split(self, df): #essa função retorna o nó raiz? de novo valor hardcodado ali
        best_gain = 0
        best_question = None
        current_impurity = self._gini(df)
        features = df.columns.drop(self.target_col)
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
            return LeafNode(df, self.target_col)
        true_rows, false_rows = self._partition(df, question)
        true_branch = self._build_tree(true_rows)
        false_branch = self._build_tree(false_rows)
        node = DecisionNode(question, true_branch, false_branch)
        return node

    def predict(self, df_test):
        predictions = []
        for _, row in df_test.iterrows():
            result_dict = self._classify(row, self.root)
            winning_class = max(result_dict, key=result_dict.get)
            predictions.append(winning_class)
        return predictions

    def _classify(self, row, node):
        if isinstance(node, LeafNode):
            return node.predictions
            
        if node.question.match(row):
            return self._classify(row, node.true_branch)
        else:
            return self._classify(row, node.false_branch)

    def fit(self, df):
        self.root = self._build_tree(df)
