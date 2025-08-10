# titanic_quick.py
"""
Quick Titanic survival classifier:
- Loads seaborn's titanic dataset
- Minimal preprocessing
- Trains logistic regression pipeline
- Prints metrics and shows confusion matrix + top features
"""
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_data():
    df = sns.load_dataset('titanic')
    print("Columns available:", list(df.columns))
    return df

def prepare_features(df):
    # Identify target
    if 'survived' in df.columns:
        target = 'survived'
    else:
        raise ValueError("Couldn't find 'survived' column in dataset")

    # Pick a compact set of features
    # pclass may be 'pclass' or 'class' in some datasets; handle both
    if 'pclass' in df.columns:
        pclass_col = 'pclass'
    elif 'class' in df.columns:
        pclass_col = 'class'
    else:
        pclass_col = None

    features = []
    for c in ['sex', 'age', 'fare', 'sibsp', 'parch', 'embarked']:
        if c in df.columns:
            features.append(c)
    if pclass_col:
        if pclass_col not in features:
            features.append(pclass_col)

    print("Using features:", features)
    df = df[features + [target]].copy()

    # Fill missing numeric with median
    for num in ['age', 'fare', 'sibsp', 'parch']:
        if num in df.columns:
            df[num] = df[num].fillna(df[num].median())

    # Fill categorical NAs with mode
    for cat in ['sex', 'embarked', pclass_col]:
        if cat and cat in df.columns:
            df[cat] = df[cat].fillna(df[cat].mode().iloc[0])

    # Drop rows if target missing
    df = df.dropna(subset=[target])

    X = df[features]
    y = df[target].astype(int)
    return X, y

def build_and_train(X_train, y_train, numeric_features, categorical_features):
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )

    clf = Pipeline(steps=[
        ('pre', preprocessor),
        ('clf', LogisticRegression(max_iter=1000))
    ])

    clf.fit(X_train, y_train)
    return clf, preprocessor

def main():
    df = load_data()
    X, y = prepare_features(df)

    # split features into numeric/categorical lists as present in X
    numeric_features = [c for c in X.columns if X[c].dtype.kind in 'biufc' and c != 'sex']
    categorical_features = [c for c in X.columns if c not in numeric_features]

    # Ensure categorical features are strings (for OneHotEncoder)
    for c in categorical_features:
        X[c] = X[c].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    clf, pre = build_and_train(X_train, y_train, numeric_features, categorical_features)

    # Predict + metrics
    y_pred = clf.predict(X_test)
    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification report:\n", classification_report(y_test, y_pred, digits=4))

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest')
    ax.set_title("Confusion matrix (rows: true, cols: pred)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.colorbar(im, ax=ax)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='white' if cm[i, j] > cm.max()/2 else 'black')
    plt.show()

    # Show top model coefficients (approx): get transformed feature names
    try:
        # get feature names from preprocessor
        feature_names = []
        # numeric names
        if numeric_features:
            feature_names.extend(numeric_features)
        # categorical names from OneHotEncoder
        cat_transformer = pre.named_transformers_['cat']
        if hasattr(cat_transformer, 'get_feature_names_out'):
            cat_names = list(cat_transformer.get_feature_names_out(categorical_features))
            feature_names.extend(cat_names)
        else:
            # fallback
            feature_names.extend(categorical_features)

        coefs = clf.named_steps['clf'].coef_[0]
        coef_df = pd.DataFrame({'feature': feature_names, 'coef': coefs})
        coef_df['abs_coef'] = coef_df['coef'].abs()
        coef_df = coef_df.sort_values('abs_coef', ascending=False).reset_index(drop=True)
        print("\nTop features by absolute coefficient:\n", coef_df.head(10).to_string(index=False))
    except Exception as e:
        print("Could not extract feature names for coefficients:", e)

if __name__ == '__main__':
    main()

