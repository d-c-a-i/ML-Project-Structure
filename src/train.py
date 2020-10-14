import os
import config
import model_dispatcher
import joblib
import pandas as pd
import argparse
from sklearn import metrics

def run(fold, model):
    df = pd.read_csv(config.TRAINING_FILE)
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_val = df[df.kfold == fold].reset_index(drop=True) 
    x_train = df_train.drop("label", axis=1).values
    y_train = df_train.label.values
    x_valid = df_val.drop("label", axis=1).values
    y_valid = df_val.label.values

    clf = model_dispatcher.models[model]
    clf.fit(x_train, y_train)
    preds = clf.predict(x_valid)

    accuracy = metrics.accuracy_score(y_valid, preds)
    print(f"Fold={fold}, Accuracy={accuracy}")

    joblib.dump(
        clf, 
        os.path.join(config.MODEL_OUTPUT, f"dt_{fold}.bin"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fold",
        type=int
    )
    parser.add_argument(
        "--model",
        type=str
    )
    args = parser.parse_args()
    # run the shell script using sh run.sh or 
    # run python -m src.train --fold 2 --model decision_tree_gini in terminal
    run(fold=args.fold,
        model=args.model
    )


