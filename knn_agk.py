import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import time
import sys

def load_data(file_path):
    return pd.read_csv(file_path, header=None, names=['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack', 'level'])

def preprocess_data(data):
    data["attack"] = data.attack.apply(lambda x: 0 if x == "normal" else 1)
    train = data[["src_bytes", "dst_bytes", "protocol_type", "attack"]]
    X = train.drop(columns="attack")
    y = train["attack"]

    label_encoder = LabelEncoder()
    X['protocol_type'] = label_encoder.fit_transform(X['protocol_type'])

    scaler = MinMaxScaler()
    X[['src_bytes', 'dst_bytes']] = scaler.fit_transform(X[['src_bytes', 'dst_bytes']])

    return X, y

def train_evaluate_model(X_train, X_test, y_train, y_test, param_grid):
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, param_grid, cv=3, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    best_knn = KNeighborsClassifier(n_neighbors=best_params['n_neighbors'])
    best_knn.fit(X_train, y_train)

    y_pred = best_knn.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return best_knn, y_pred, conf_matrix, accuracy, report

def calculate_complexity(train_data, X_train, X_test, y_train, y_test, grid_search, best_knn, y_pred):
    space_complexity = {
        'train_data': sys.getsizeof(train_data),
        'X_train': sys.getsizeof(X_train),
        'X_test': sys.getsizeof(X_test),
        'y_train': sys.getsizeof(y_train),
        'y_test': sys.getsizeof(y_test),
        'grid_search': sys.getsizeof(grid_search),
        'best_knn': sys.getsizeof(best_knn),
        'y_pred': sys.getsizeof(y_pred)
    }
    total_space_complexity = sum(space_complexity.values())
    return space_complexity, total_space_complexity

def train_and_evaluate(file_path, use_pca=False):
    start_time = time.time()

    # Loading data
    train_data = load_data(file_path)

    # Preprocessing data
    X, y = preprocess_data(train_data)

    # Applying PCA if required
    if use_pca:
        pca = PCA(n_components=2)
        X = pca.fit_transform(X)

    # Splitting data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Defining parameter grid for hyperparameter tuning
    param_grid = {'n_neighbors': [3, 5, 7, 9]}

    # Training and evaluating model
    best_knn, y_pred, conf_matrix, accuracy, report = train_evaluate_model(X_train, X_test, y_train, y_test, param_grid)

    # Calculating complexity
    space_complexity, total_space_complexity = calculate_complexity(train_data, X_train, X_test, y_train, y_test, None, best_knn, y_pred)

    # Printing results
    print("Best Hyperparameters:", best_knn.get_params())
    print("Confusion Matrix:")
    print(conf_matrix)
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, cmap='Blues', interpolation='nearest')
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(report)
    for obj, size in space_complexity.items():
        print(f"Space Complexity of {obj}: {size} bytes")
    print(f"Total Space Complexity: {total_space_complexity} bytes")

    # Recording end time
    end_time = time.time()

    # Calculating execution time
    execution_time = end_time - start_time
    print("Execution Time:", execution_time, "seconds")

if __name__ == "__main__":
    train_and_evaluate("C:\\Users\\kaksh\\Desktop\\knn\\data\\KDDTest+.txt", use_pca=True)
    train_and_evaluate("C:\\Users\\kaksh\\Desktop\\knn\\data\\KDDTest+.txt", use_pca=False)  
