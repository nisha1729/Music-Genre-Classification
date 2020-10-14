######################
# 1.3 Data Pre-Processing
######################

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def data_process():

    feat = pd.read_csv('extracted_features.csv')
    feat = feat.drop(['filename'],axis=1)   # Remove irrelevant features
    feat = feat.iloc[:-1, :]                # Remove last NaN line
    genre_list = feat.iloc[:, -1]           # Extract genre list

    # Encode string labels as numbers
    encoder = LabelEncoder()
    y = encoder.fit_transform(genre_list.astype(str))

    # Normalise features
    scaler = StandardScaler()
    X = scaler.fit_transform(np.array(feat.iloc[:, :-1], dtype = float))

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = data_process()