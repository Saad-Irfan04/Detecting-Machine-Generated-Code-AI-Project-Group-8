from data_loader import load_dataset
from preprocess import preprocess
from model_baseline_A import BaselineA

def main():
    df = load_dataset("data/dataset.csv")
    X_train, y_train, X_test, y_test = preprocess(df)

    model = BaselineA()
    model.train(X_train, y_train)

if __name__ == "__main__":
    main()
