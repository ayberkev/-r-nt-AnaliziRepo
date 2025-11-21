import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

# Deep learning imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, LSTM, Dropout

RAW_GITHUB_URL = "https://raw.githubusercontent.com/ayberkev/-r-nt-AnaliziRepo/main/TEIAS_Elektrik_Tuketim_Verileri_2015_2025.csv"

def load_data():
    data_set = pd.read_csv(RAW_GITHUB_URL)
    data_set_classes = data_set['Önerilen_Enerji_Türü'].astype('category').cat.codes
    data_set_attributes = data_set[['Toplam_Üretim_MWh','Toplam_Tüketim_MWh','Yenilenebilir_Oranı','Pik_Talep_MW']]
    n_classes = len(data_set['Önerilen_Enerji_Türü'].unique())
    return data_set_attributes, data_set_classes, n_classes

def preprocess_data(data_set_attributes):
    data_set_attributes = data_set_attributes.select_dtypes(include=[np.number])
    scaler = StandardScaler()
    return scaler.fit_transform(data_set_attributes)

def apply_pca(data_set_attributes_scaled, n_components=0.95, use_pca=False):
    if use_pca:
        pca = PCA(n_components=n_components)
        return pca.fit_transform(data_set_attributes_scaled)
    return data_set_attributes_scaled
    
def apply_lda(data_set_attributes_scaled, data_set_classes, n_classes, use_lda=False):
    n_components = min(data_set_attributes_scaled.shape[1], n_classes - 1)
    if use_lda:
        lda = LDA(n_components=n_components)
        return lda.fit_transform(data_set_attributes_scaled, data_set_classes)
    return data_set_attributes_scaled

def evaluate_model(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    accuracy = np.mean(y_pred == y_true)
    sensivity = np.diag(cm) / np.sum(cm, axis=1)
    sensivity = np.mean(sensivity[np.isfinite(sensivity)])
    specivity = 1 - sensivity
    f1_score = (2 * sensivity * specivity) / (sensivity + specivity) if (sensivity + specivity) != 0 else 0
    auc_roc = 0.5
    return accuracy, sensivity, specivity, f1_score, auc_roc

def build_cnn(input_shape, n_classes):
    model = Sequential([
        Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(n_classes, activation='softmax')
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model
    
def build_lstm(input_shape, n_classes):
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=False),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(n_classes, activation='softmax')
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def plot_confusion_matrix(y_true, y_pred, model_name, class_labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_labels, yticklabels=class_labels)
    plt.savefig(f"{model_name}_confusion_matrix.png")
    plt.close()

def main(use_pca=False, use_lda=False):
    data_set_attributes, data_set_classes, n_classes = load_data()
    original_classes = pd.read_csv(RAW_GITHUB_URL)['Önerilen_Enerji_Türü'].astype('category').cat.categories.tolist()
    X_scaled = preprocess_data(data_set_attributes)

    if use_pca: X = apply_pca(X_scaled, use_pca=True)
    elif use_lda: X = apply_lda(X_scaled, data_set_classes, n_classes, use_lda=True)
    else: X = X_scaled

    y = data_set_classes.values
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "1D-CNN": "CNN",
        "LSTM": "LSTM"
    }

    results = []

    for model_name, model in models.items():
        fold_results = {k: [] for k in ["acc", "sens", "spec", "f1", "auc"]}
        all_true, all_pred = [], []

        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            if model_name in ["1D-CNN", "LSTM"]:
                X_train_dl = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
                X_test_dl = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

                dl_model = build_cnn(X_train_dl.shape[1:], n_classes) if model_name == "1D-CNN" else build_lstm(X_train_dl.shape[1:], n_classes)
                dl_model.fit(X_train_dl, y_train, epochs=25, batch_size=32, verbose=0)
                y_pred = np.argmax(dl_model.predict(X_test_dl, verbose=0), axis=1)

            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

            accuracy, sensivity, specivity, f1, auc = evaluate_model(y_test, y_pred)
            fold_results["acc"].append(accuracy)
            fold_results["sens"].append(sensivity)
            fold_results["spec"].append(specivity)
            fold_results["f1"].append(f1)
            fold_results["auc"].append(auc)

            all_true.extend(y_test)
            all_pred.extend(y_pred)

        results.append({
            "Model": model_name,
            "Accuracy": np.mean(fold_results["acc"]),
            "Sensivity": np.mean(fold_results["sens"]),
            "Specivity": np.mean(fold_results["spec"]),
            "F-Measure": np.mean(fold_results["f1"]),
            "AUC-ROC": np.mean(fold_results["auc"])
        })

        plot_confusion_matrix(np.array(all_true), np.array(all_pred), model_name, original_classes)

    print(pd.DataFrame(results))


if __name__ == '__main__':
    main(use_pca=False, use_lda=False)
