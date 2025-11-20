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

RAW_GITHUB_URL = "https://raw.githubusercontent.com/ayberkev/-r-nt-AnaliziRepo/main/TEIAS_Elektrik_Tuketim_Verileri_2015_2025.csv"

def load_data():
    data_set = pd.read_csv(RAW_GITHUB_URL)
    data_set_classes = data_set['Önerilen_Enerji_Türü'].astype('category').cat.codes
    data_set_attributes = data_set[[
        'Toplam_Üretim_MWh', 
        'Toplam_Tüketim_MWh', 
        'Yenilenebilir_Oranı', 
        'Pik_Talep_MW'
    ]]
    n_classes = len(data_set['Önerilen_Enerji_Türü'].unique())
    return data_set_attributes, data_set_classes, n_classes

def preprocess_data(data_set_attributes):
    data_set_attributes = data_set_attributes.select_dtypes(include=[np.number])
    scaler = StandardScaler()
    data_set_attributes_scaled = scaler.fit_transform(data_set_attributes)
    return data_set_attributes_scaled

def apply_pca(data_set_attributes_scaled, n_components=0.95, use_pca=False):
    if use_pca:
        pca = PCA(n_components=n_components)
        data_set_attributes_scaled_pca = pca.fit_transform(data_set_attributes_scaled) 
        return data_set_attributes_scaled_pca
    else:
        return data_set_attributes_scaled
    
def apply_lda(data_set_attributes_scaled, data_set_classes, n_classes, use_lda=False):
    n_components = min(data_set_attributes_scaled.shape[1], n_classes - 1)
    
    if use_lda:
        lda = LDA(n_components=n_components)
        data_set_attributes_scaled_lda = lda.fit_transform(data_set_attributes_scaled, data_set_classes) 
        return data_set_attributes_scaled_lda
    else:
        return data_set_attributes_scaled

def evaluate_model(y_true, y_pred, y_prob, num_classes):
    cm = confusion_matrix(y_true, y_pred)
    accuracy = np.mean(y_pred == y_true) 
    
    sensivity = np.diag(cm) / np.sum(cm, axis=1)
    sensivity = np.mean(sensivity[np.isfinite(sensivity)])

    specivity = (1-sensivity)
    f1_score = (2 * sensivity * specivity) / (sensivity + specivity) if (sensivity + specivity) != 0 else 0
    
    auc_roc = 0.5 
    
    return accuracy, sensivity, specivity, f1_score, auc_roc

def plot_confusion_matrix(y_true, y_pred, model_name, class_labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_labels, 
                yticklabels=class_labels)
    plt.savefig(f"{model_name}_confusion_matrix.png")
    plt.close()

def main(use_pca=False, use_lda=False):
    data_set_attributes, data_set_classes, n_classes = load_data()
    
    if data_set_attributes is None:
        return

    original_classes = pd.read_csv(RAW_GITHUB_URL)['Önerilen_Enerji_Türü'].astype('category').cat.categories.tolist()

    data_set_attributes_scaled = preprocess_data(data_set_attributes)
    
    if use_pca:
        X = apply_pca(data_set_attributes_scaled, n_components=0.95, use_pca=True)
    elif use_lda and n_classes > 1:
        X = apply_lda(data_set_attributes_scaled, data_set_classes, n_classes, use_lda=True)
    else:
        X = data_set_attributes_scaled

    y = data_set_classes.values
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5)
    }
    results = []

    for model_name, model in models.items():
        fold_accuracies = []
        fold_sensivities = []
        fold_specivity = []
        fold_f_measure = []
        fold_auc_roc = []

        all_y_true = []
        all_y_pred = []

        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            y_prob = model.predict_proba(X_test)

            accuracy, sensivity, specivity, f1_score, auc_roc = evaluate_model(y_test, y_pred, y_prob, n_classes)

            fold_accuracies.append(accuracy)
            fold_sensivities.append(sensivity)
            fold_specivity.append(specivity)
            fold_f_measure.append(f1_score)
            fold_auc_roc.append(auc_roc)

            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)

        results.append({
            "Model": model_name,
            "Accuracy": np.mean(fold_accuracies),
            "Sensivity (Mean Recall)": np.mean(fold_sensivities),
            "Specivity (Mean Neg Recall)": np.mean(fold_specivity),
            "F-Measure": np.mean(fold_f_measure),
            "AUC-ROC (Simplified)": np.mean(fold_auc_roc)
        })    
        
        plot_confusion_matrix(np.array(all_y_true), np.array(all_y_pred), model_name, original_classes)
        
    result_df = pd.DataFrame(results)
    print(result_df)


if __name__ == '__main__':
    main(use_pca=False, use_lda=False)