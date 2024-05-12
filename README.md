# IDS-using-KNN
Anomaly Detection in Network Security using KNN and a Comparative Study
Author: Akshay Gangadhar Kanamarlapudi

#Introduction
Anomaly detection is a critical aspect of network security, aiming to identify patterns or instances that deviate significantly from expected behavior within a network. It plays a crucial role in maintaining network security by detecting malicious activities, intrusions, or abnormal behavior.

#Challenges in Detecting Anomalies in Network Traffic Data
High-dimensional data: Network traffic data often contain numerous features, making it challenging to identify anomalies.
Imbalanced class distribution: The majority of network traffic is normal, while anomalies (attacks) are relatively rare.
Evolving attack techniques: Attackers constantly adapt, necessitating robust detection methods.

#Dataset Description
We utilize the popular KDD Cup dataset for intrusion detection systems, containing network traffic data collected from a simulated military network environment. The dataset includes two files: KDDTrain+.txt (training data) and KDDTest+.txt (testing data). We split the training data into 80% for training and 20% for testing.
The dataset comprises 41 network connection records, with each row representing a connection and features such as protocol type, source/destination bytes, flags, error rates, etc., indicating whether the connection is an attack or normal.

#Relevant Features
protocol_type
src_bytes
dst_bytes

#K-Nearest Neighbors (KNN) and Its Methodology
#What is KNN?
K-Nearest Neighbors (KNN) is a simple yet powerful algorithm for classification tasks. It classifies data points based on their similarity to their nearest neighbors.

#How Does It Work?
Calculate the Euclidean distance between data points.
Choose an appropriate value for K (number of neighbors).
The majority vote among K neighbors determines the class label.
#When to Use KNN?
Properly labeled data.
Noise-free data.
Small datasets.
Data Preprocessing Steps
Encoding categorical features (protocol_type).
Scaling numerical features (src_bytes, dst_bytes).
Train-test split for model evaluation.
Hyperparameter tuning using GridSearchCV.
Predict on test data.

#KNN Results
We identified the optimal hyperparameters, generated a confusion matrix, calculated accuracy, and the classification report, describing the results of using KNN on the dataset.

#Principal Component Analysis (PCA)
Principal Component Analysis (PCA) is a dimensionality reduction technique commonly used in data preprocessing and feature extraction. It transforms high-dimensional data into a lower-dimensional space while retaining the most important information. We observed consistent outcomes when comparing results obtained with and without PCA for our dataset.

#Computational Complexity - KNN
#Time complexity: O(n * k * d)
n: number of samples
k: number of neighbors
d: number of features
#Space complexity: O(n * d)
Explanation: KNN has a time complexity that scales linearly with the number of samples, the number of neighbors, and the number of features. The space complexity scales linearly with the number of samples and features.

#Potential Improvements
Feature engineering: Explore additional relevant features or feature transformations that could potentially improve the model's performance.
Advanced hyperparameter tuning techniques: Implement techniques like random search, Bayesian optimization, or genetic algorithms for more efficient hyperparameter tuning.
Handling class imbalance: Investigate techniques like oversampling (e.g., SMOTE) or undersampling to address the class imbalance issue.
Ensemble or hybrid approaches: Explore combining multiple methods (e.g., KNN and Isolation Forest) to leverage their strengths and mitigate their weaknesses.

#Conclusion
Anomaly detection involves identifying patterns or instances in a dataset that significantly deviate from expected behavior. Among various techniques, K-Nearest Neighbors (KNN) stands out as effective for local patterns but sensitive to outliers. Researchers recommend hybrid approaches, feature engineering, and advanced hyperparameter tuning to enhance anomaly detection performance. Additionally, exploring deep learning models and collaborating with domain experts can lead to further improvements.
