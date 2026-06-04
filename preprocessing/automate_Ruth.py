import os
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def preprocessing():

    # Load dataset
    df = pd.read_csv(
        "../dataset_raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )

    print("Dataset berhasil dimuat")

    # Hapus customerID
    df.drop("customerID", axis=1, inplace=True)

    # Ubah TotalCharges menjadi numerik
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Hapus missing value
    df.dropna(inplace=True)

    print("Missing value berhasil ditangani")

    # Encoding
    le = LabelEncoder()

    for col in df.select_dtypes(
        include="object"
    ).columns:
        df[col] = le.fit_transform(df[col])

    print("Encoding selesai")

    # Pisahkan feature dan target
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print("Split dataset selesai")

    # Scaling
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Ubah ke DataFrame
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)

    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)

    # Membuat folder output
    os.makedirs(
        "dataset_preprocessing",
        exist_ok=True
    )

    # Simpan hasil preprocessing
    X_train.to_csv(
        "dataset_preprocessing/X_train.csv",
        index=False
    )

    X_test.to_csv(
        "dataset_preprocessing/X_test.csv",
        index=False
    )

    y_train.to_csv(
        "dataset_preprocessing/y_train.csv",
        index=False
    )

    y_test.to_csv(
        "dataset_preprocessing/y_test.csv",
        index=False
    )

    print("Dataset preprocessing berhasil disimpan")


if __name__ == "__main__":
    preprocessing()