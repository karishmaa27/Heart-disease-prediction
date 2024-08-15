import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Custom SVM Classifier
class SVM_Classifier():
    def __init__(self, learning_rate, no_of_iterations, lambda_parameter, kernel='linear', degree=3, gamma=None, coef0=1):
        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations
        self.lambda_parameter = lambda_parameter
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0

    def fit(self, X, Y):
        self.m, self.n = X.shape
        self.w = np.zeros(self.m)
        self.b = 0
        self.X = X
        self.Y = np.where(Y <= 0, -1, 1)
        self.gamma = self.gamma if self.gamma is not None else 1 / self.n

        self.K = self.compute_kernel_matrix(self.X)

        for _ in range(self.no_of_iterations):
            self.update_weight()

    def compute_kernel_matrix(self, X):
        m = X.shape[0]
        K = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                K[i, j] = self.kernel_function(X[i], X[j])
        return K

    def kernel_function(self, x1, x2):
        if self.kernel == 'linear':
            return np.dot(x1, x2)
        elif self.kernel == 'sigmoid':
            return np.tanh(self.gamma * np.dot(x1, x2) + self.coef0)
        elif self.kernel == 'rbf':
            return np.exp(-self.gamma * np.linalg.norm(x1 - 1.2*x2) ** 2)
        elif self.kernel == 'polynomial':
            return (self.gamma * np.dot(x1, x2) + self.coef0) ** self.degree
        elif self.kernel == 'custom':
            rbf_part = np.exp(-self.gamma * np.linalg.norm(0.5*x1 - 0.9*x2) ** 3)
            poly_part = (self.gamma * np.dot(x1, 0.9*x2) + self.coef0) ** self.degree
            return rbf_part * poly_part
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel}")

    def update_weight(self):
        y_label = np.where(self.Y <= 0, -1, 1)
        for index in range(self.m):
            decision_value = np.dot(self.K[index], self.w) - self.b
            condition = y_label[index] * decision_value >= 1

            if np.any(condition):
                dw = 2 * self.lambda_parameter * self.w
                db = 0
            else:
                dw = 2 * self.lambda_parameter * self.w - y_label[index] * self.K[index]
                db = y_label[index]

            self.w = self.w - self.learning_rate * dw
            self.b = self.b - self.learning_rate * db

    def predict(self, X):
        m = X.shape[0]
        K = np.zeros((m, self.m))
        for i in range(m):
            for j in range(self.m):
                K[i, j] = self.kernel_function(X[i], self.X[j])

        output = np.dot(K, self.w) - self.b
        predicted_labels = np.sign(output)
        y_hat = np.where(predicted_labels <= -1, 0, 1)
        return y_hat

# Load and preprocess the dataset
heart_data = pd.read_csv('heart_data.csv')
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

# Standardizing the data
scaler = StandardScaler()
X=scaler.fit_transform(X)

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,random_state=2)

# Initialize the SVM classifier
classifier = SVM_Classifier(learning_rate=0.001, no_of_iterations=5000, lambda_parameter=0.01, kernel='custom')

# Train the classifier
classifier.fit(X_train,Y_train)

# Test the classifier
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)

print("Accuracy score on testing data: ", test_data_accuracy)

# Save the trained model
pickle.dump(classifier, open('heart_svm.pkl', 'wb'))
