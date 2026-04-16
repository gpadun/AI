import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from src.decision_tree import DecisionTree

def main():
    df = pd.read_csv('dataset/IRIS.csv')

    df_train, df_test = train_test_split(
        df,
        test_size=0.2,
        stratify=df['species'],
        random_state=42
    )

    print(f"Training data: {len(df_train)} rows")
    print(f"Test data: {len(df_test)} rows\n")

    model = DecisionTree()
    model.fit(df_train)

    predictions = model.predict(df_test)
    y_true = df_test['species'].to_list()

    accuracy = accuracy_score(y_true, predictions)
    print(f"Accuracy: {accuracy * 100:.2f}%\n")

    print("Classification Report:")
    print(classification_report(y_true, predictions))

if __name__ == "__main__":
    main()
