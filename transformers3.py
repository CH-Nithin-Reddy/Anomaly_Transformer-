# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import pandas as pd
# import numpy as np
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# # Load and preprocess data
# df = pd.read_csv('transformerfront/btc_1d_data_2018_to_2024-2024-12-10.csv')
# df = df.drop(['Open time', 'Close time'], axis=1)
# X = df.values
# X_train = torch.FloatTensor(X)

# # Normalize data
# X_train = (X_train - X_train.mean(dim=0)) / (X_train.std(dim=0) + 1e-8)  # Prevent division by zero
# X_train = torch.nan_to_num(X_train, nan=0.0, posinf=1.0, neginf=-1.0)    # Replace NaN/inf values

# class AnomalyAttention(nn.Module):
#     def __init__(self, N, d_model):
#         super(AnomalyAttention, self).__init__()
#         self.N = N
#         self.d_model = d_model
#         self.Wq = nn.Linear(d_model, d_model, bias=False)
#         self.Wk = nn.Linear(d_model, d_model, bias=False)
#         self.Wv = nn.Linear(d_model, d_model, bias=False)
#         self.Ws = nn.Linear(d_model, 1, bias=False)

#     def forward(self, x):
#         Q = self.Wq(x)
#         K = self.Wk(x)
#         V = self.Wv(x)
#         sigma = torch.clamp(self.Ws(x), min=1e-3, max=1.0)  # Clamp sigma to avoid instability
#         P = self.prior_association(sigma)
#         S = (Q @ K.T) / torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
#         S = torch.softmax(S, dim=-1)
#         Z = S @ V
#         return Z, P, S

#     @staticmethod
#     def prior_association(sigma):
#         N = sigma.shape[0]
#         p = torch.arange(N).unsqueeze(0).repeat(N, 1)
#         diff = torch.abs(p - p.T).float()
#         gaussian = torch.exp(-0.5 * (diff / sigma).pow(2)) / torch.sqrt(2 * torch.pi * sigma)
#         return gaussian / (gaussian.sum(dim=1, keepdim=True) + 1e-8)

# class AnomalyTransformer(nn.Module):
#     def __init__(self, N, d_model, hidden_dim, lambda_=0.1):
#         super(AnomalyTransformer, self).__init__()
#         self.N = N
#         self.d_model = d_model
#         self.lambda_ = lambda_
#         self.hidden_dim = hidden_dim
#         self.attention_layers = AnomalyAttention(N, d_model)
#         self.hidden_layer = nn.Linear(d_model, hidden_dim)
#         self.output_layer = nn.Linear(hidden_dim, d_model)

#     def forward(self, x):
#         Z, P, S = self.attention_layers(x)
#         hidden = torch.relu(self.hidden_layer(Z))
#         x_hat = self.output_layer(hidden)
#         return x_hat, P, S

#     def loss_function(self, x_hat, x, P, S):
#         frob_norm = torch.linalg.norm(x_hat - x, ord='fro')
#         kl_div = torch.sum(F.kl_div(torch.log(P + 1e-8), S, reduction='batchmean'))
#         return frob_norm + self.lambda_ * kl_div

#     def anomaly_score(self, x):
#         x_hat, P, S = self(x)
#         reconstruction_error = torch.linalg.norm(x - x_hat, dim=1)
#         assoc_discrepancy = torch.sum(F.kl_div(torch.log(P + 1e-8), S, reduction='batchmean'))
#         return reconstruction_error + assoc_discrepancy

# # Training loop
# def train(model, data, optimizer, epochs):
#     model.train()
#     for epoch in range(epochs):
#         optimizer.zero_grad()
#         x_hat, P, S = model(data)
#         loss = model.loss_function(x_hat, data, P, S)
#         if not torch.isfinite(loss):
#             print(f"Epoch {epoch + 1}/{epochs}, Loss encountered NaN or Inf. Skipping...")
#             continue
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
#         optimizer.step()
#         print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

# # Initialize model
# N, d_model = X_train.shape
# hidden_dim = 64
# model = AnomalyTransformer(N, d_model, hidden_dim)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# epochs = 10

# # Train the model
# train(model, X_train, optimizer, epochs)

# # Calculate anomaly scores
# anomaly_scores = model.anomaly_score(X_train).detach().numpy()
# anomaly_scores = np.nan_to_num(anomaly_scores, nan=0.0, posinf=1.0, neginf=-1.0)

# # Simulate labels
# np.random.seed(42)
# true_labels = np.zeros(N, dtype=int)
# anomaly_indices = np.random.choice(N, size=int(0.1 * N), replace=False)
# true_labels[anomaly_indices] = 1

# # Threshold and normalize scores
# threshold = np.percentile(anomaly_scores, 90)
# predictions = (anomaly_scores > threshold).astype(int)

# # Evaluate metrics
# accuracy = accuracy_score(true_labels, predictions)
# precision = precision_score(true_labels, predictions, zero_division=0)
# recall = recall_score(true_labels, predictions, zero_division=0)
# f1 = f1_score(true_labels, predictions, zero_division=0)
# roc_auc = roc_auc_score(true_labels, anomaly_scores)

# # Print metrics
# print(f"Accuracy: {accuracy * 100:.2f}%")
# print(f"Precision: {precision * 100:.2f}%")
# print(f"Recall: {recall * 100:.2f}%")
# print(f"F1 Score: {f1 * 100:.2f}%")
# print(f"ROC-AUC: {roc_auc:.4f}")

#Correct import
# import json
# import sys
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import pandas as pd
# import numpy as np
# import os
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# # Load and preprocess data
# if len(sys.argv) < 2:
#     print("Error: No file path provided.")
#     sys.exit(1)

# input_file = sys.argv[1]

# # Validate file existence
# if not os.path.exists(input_file):
#     print(f"Error: The file '{input_file}' does not exist.")
#     sys.exit(1)

# # Load and process the data
# df = pd.read_csv(input_file, delimiter=',', on_bad_lines='skip')
# df = df.apply(pd.to_numeric, errors='coerce')
# df = df.fillna(df.mean())

# X = df.values
# X_train = torch.FloatTensor(X)

# # Normalize data
# X_train = (X_train - X_train.mean(dim=0)) / (X_train.std(dim=0) + 1e-8)
# X_train = torch.nan_to_num(X_train, nan=0.0, posinf=1.0, neginf=-1.0)

# class AnomalyAttention(nn.Module):
#     def __init__(self, N, d_model):
#         super(AnomalyAttention, self).__init__()
#         self.N = N
#         self.d_model = d_model
#         self.Wq = nn.Linear(d_model, d_model, bias=False)
#         self.Wk = nn.Linear(d_model, d_model, bias=False)
#         self.Wv = nn.Linear(d_model, d_model, bias=False)
#         self.Ws = nn.Linear(d_model, 1, bias=False)

#     def forward(self, x):
#         Q = self.Wq(x)
#         K = self.Wk(x)
#         V = self.Wv(x)
#         sigma = torch.clamp(self.Ws(x), min=1e-3, max=1.0)
#         P = self.prior_association(sigma)
#         S = (Q @ K.T) / torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
#         S = torch.softmax(S, dim=-1)
#         Z = S @ V
#         return Z, P, S

#     @staticmethod
#     def prior_association(sigma):
#         N = sigma.shape[0]
#         p = torch.arange(N).unsqueeze(0).repeat(N, 1)
#         diff = torch.abs(p - p.T).float()
#         gaussian = torch.exp(-0.5 * (diff / sigma).pow(2)) / torch.sqrt(2 * torch.pi * sigma)
#         return gaussian / (gaussian.sum(dim=1, keepdim=True) + 1e-8)

# class AnomalyTransformer(nn.Module):
#     def __init__(self, N, d_model, hidden_dim, lambda_=0.1):
#         super(AnomalyTransformer, self).__init__()
#         self.N = N
#         self.d_model = d_model
#         self.lambda_ = lambda_
#         self.hidden_dim = hidden_dim
#         self.attention_layers = AnomalyAttention(N, d_model)
#         self.hidden_layer = nn.Linear(d_model, hidden_dim)
#         self.output_layer = nn.Linear(hidden_dim, d_model)

#     def forward(self, x):
#         Z, P, S = self.attention_layers(x)
#         hidden = torch.relu(self.hidden_layer(Z))
#         x_hat = self.output_layer(hidden)
#         return x_hat, P, S

#     def loss_function(self, x_hat, x, P, S):
#         frob_norm = torch.linalg.norm(x_hat - x, ord='fro')
#         kl_div = torch.sum(F.kl_div(torch.log(P + 1e-8), S, reduction='batchmean'))
#         return frob_norm + self.lambda_ * kl_div

#     def anomaly_score(self, x):
#         x_hat, P, S = self(x)
#         reconstruction_error = torch.linalg.norm(x - x_hat, dim=1)
#         assoc_discrepancy = torch.sum(F.kl_div(torch.log(P + 1e-8), S, reduction='batchmean'))
#         return reconstruction_error + assoc_discrepancy

# def train(model, data, optimizer, epochs):
#     model.train()
#     total_loss = 0.0
#     for epoch in range(epochs):
#         optimizer.zero_grad()
#         x_hat, P, S = model(data)
#         loss = model.loss_function(x_hat, data, P, S)
#         if not torch.isfinite(loss):
#             print(f"Epoch {epoch + 1}/{epochs}, Loss encountered NaN or Inf. Skipping...")
#             continue
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#         optimizer.step()
#         total_loss += loss.item()
#     return total_loss / epochs  # Return the average loss over epochs

# N, d_model = X_train.shape
# hidden_dim = 64
# model = AnomalyTransformer(N, d_model, hidden_dim)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# epochs = 10

# # Train the model
# average_loss = train(model, X_train, optimizer, epochs)

# # Calculate anomaly scores
# anomaly_scores = model.anomaly_score(X_train).detach().numpy()
# anomaly_scores = np.nan_to_num(anomaly_scores, nan=0.0, posinf=1.0, neginf=-1.0)

# # Simulate labels
# np.random.seed(42)
# true_labels = np.zeros(N, dtype=int)
# anomaly_indices = np.random.choice(N, size=int(0.1 * N), replace=False)
# true_labels[anomaly_indices] = 1

# # Threshold and normalize scores
# threshold = np.percentile(anomaly_scores, 90)
# predictions = (anomaly_scores > threshold).astype(int)

# # Get the indices of anomalies (where prediction is 1)
# anomaly_indices_detected = np.where(predictions == 1)[0].tolist()

# # Evaluate metrics
# accuracy = accuracy_score(true_labels, predictions)
# precision = precision_score(true_labels, predictions, zero_division=0)
# recall = recall_score(true_labels, predictions, zero_division=0)
# f1 = f1_score(true_labels, predictions, zero_division=0)
# roc_auc = roc_auc_score(true_labels, anomaly_scores)

# # Create the result JSON
# result = {
#     "anomalies": anomaly_indices_detected,  # Indices of anomalies
#     "losses": [average_loss] * epochs,  # Or just return [average_loss] or the list of losses
#     "threshold": float(threshold),
#     "combined_scores": anomaly_scores.tolist()
# }


# # Output the result as JSON
# output_json = json.dumps(result, indent=4)
# print(output_json)  # This will be captured by the backend



#Prints loss and accuracy(using torch)
# import json
# import sys
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import pandas as pd
# import numpy as np
# import os
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# # Load and preprocess data
# if len(sys.argv) < 2:
#     print("Error: No file path provided.")
#     sys.exit(1)

# input_file = sys.argv[1]

# # Validate file existence
# if not os.path.exists(input_file):
#     print(f"Error: The file '{input_file}' does not exist.")
#     sys.exit(1)

# # Load and process the data
# df = pd.read_csv(input_file, delimiter=',', on_bad_lines='skip')
# df = df.apply(pd.to_numeric, errors='coerce')
# df = df.fillna(df.mean())

# X = df.values
# X_train = torch.FloatTensor(X)

# # Normalize data
# X_train = (X_train - X_train.mean(dim=0)) / (X_train.std(dim=0) + 1e-8)
# X_train = torch.nan_to_num(X_train, nan=0.0, posinf=1.0, neginf=-1.0)

# class AnomalyAttention(nn.Module):
#     def __init__(self, N, d_model):
#         super(AnomalyAttention, self).__init__()
#         self.N = N
#         self.d_model = d_model
#         self.Wq = nn.Linear(d_model, d_model, bias=False)
#         self.Wk = nn.Linear(d_model, d_model, bias=False)
#         self.Wv = nn.Linear(d_model, d_model, bias=False)
#         self.Ws = nn.Linear(d_model, 1, bias=False)

#     def forward(self, x):
#         Q = self.Wq(x)
#         K = self.Wk(x)
#         V = self.Wv(x)
#         sigma = torch.clamp(self.Ws(x), min=1e-3, max=1.0)
#         P = self.prior_association(sigma)
#         S = (Q @ K.T) / torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
#         S = torch.softmax(S, dim=-1)
#         Z = S @ V
#         return Z, P, S

#     @staticmethod
#     def prior_association(sigma):
#         N = sigma.shape[0]
#         p = torch.arange(N).unsqueeze(0).repeat(N, 1)
#         diff = torch.abs(p - p.T).float()
#         gaussian = torch.exp(-0.5 * (diff / sigma).pow(2)) / torch.sqrt(2 * torch.pi * sigma)
#         return gaussian / (gaussian.sum(dim=1, keepdim=True) + 1e-8)

# class AnomalyTransformer(nn.Module):
#     def __init__(self, N, d_model, hidden_dim, lambda_=0.1):
#         super(AnomalyTransformer, self).__init__()
#         self.N = N
#         self.d_model = d_model
#         self.lambda_ = lambda_
#         self.hidden_dim = hidden_dim
#         self.attention_layers = AnomalyAttention(N, d_model)
#         self.hidden_layer = nn.Linear(d_model, hidden_dim)
#         self.output_layer = nn.Linear(hidden_dim, d_model)

#     def forward(self, x):
#         Z, P, S = self.attention_layers(x)
#         hidden = torch.relu(self.hidden_layer(Z))
#         x_hat = self.output_layer(hidden)
#         return x_hat, P, S

#     def loss_function(self, x_hat, x, P, S):
#         frob_norm = torch.linalg.norm(x_hat - x, ord='fro')
#         kl_div = torch.sum(F.kl_div(torch.log(P + 1e-8), S, reduction='batchmean'))
#         return frob_norm + self.lambda_ * kl_div

#     def anomaly_score(self, x):
#         x_hat, P, S = self(x)
#         reconstruction_error = torch.linalg.norm(x - x_hat, dim=1)
#         assoc_discrepancy = torch.sum(F.kl_div(torch.log(P + 1e-8), S, reduction='batchmean'))
#         return reconstruction_error + assoc_discrepancy

# def train(model, data, optimizer, epochs):
#     model.train()
#     total_loss = 0.0
#     for epoch in range(epochs):
#         optimizer.zero_grad()
#         x_hat, P, S = model(data)
#         loss = model.loss_function(x_hat, data, P, S)
#         if not torch.isfinite(loss):
#             print(f"Epoch {epoch + 1}/{epochs}, Loss encountered NaN or Inf. Skipping...")
#             continue
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#         optimizer.step()
        
#         # Print the loss for the current epoch
#         print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")
#         total_loss += loss.item()
#     return total_loss / epochs  # Return the average loss over epochs

# N, d_model = X_train.shape
# hidden_dim = 64
# model = AnomalyTransformer(N, d_model, hidden_dim)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# epochs = 10

# # Train the model
# average_loss = train(model, X_train, optimizer, epochs)

# # Calculate anomaly scores
# anomaly_scores = model.anomaly_score(X_train).detach().numpy()
# anomaly_scores = np.nan_to_num(anomaly_scores, nan=0.0, posinf=1.0, neginf=-1.0)

# # Apply minimax strategy to refine the anomaly detection threshold
# minimax_scores = []
# for i, score in enumerate(anomaly_scores):
#     minimax_scores.append(max(score, min(anomaly_scores)))

# # Simulate labels
# np.random.seed(42)
# true_labels = np.zeros(N, dtype=int)
# anomaly_indices = np.random.choice(N, size=int(0.1 * N), replace=False)
# true_labels[anomaly_indices] = 1

# # Threshold and normalize scores
# threshold = np.percentile(minimax_scores, 90)
# predictions = (minimax_scores > threshold).astype(int)

# # Get the indices of anomalies (where prediction is 1)
# anomaly_indices_detected = np.where(predictions == 1)[0].tolist()

# # Evaluate metrics
# accuracy = accuracy_score(true_labels, predictions)
# precision = precision_score(true_labels, predictions, zero_division=0)
# recall = recall_score(true_labels, predictions, zero_division=0)
# f1 = f1_score(true_labels, predictions, zero_division=0)
# roc_auc = roc_auc_score(true_labels, minimax_scores)



# # Create the result JSON
# result = {
#     "anomalies": anomaly_indices_detected,  # Indices of anomalies
#     "losses": [average_loss] * epochs,  # Or just return [average_loss] or the list of losses
#     "threshold": float(threshold),
#     "combined_scores": anomaly_scores.tolist()
# }

# # Print metrics
# # print(f"Accuracy: {accuracy * 100:.2f}%")
# # print(f"Precision: {precision * 100:.2f}%")
# # print(f"Recall: {recall * 100:.2f}%")
# # print(f"F1 Score: {f1 * 100:.2f}%")
# # print(f"ROC-AUC: {roc_auc:.4f}")
# # Output the result as JSON
# output_json = json.dumps(result, indent=4)
# print(output_json)


# import pandas as pd
# import numpy as np
# import os
# import sys
# from sklearn.preprocessing import StandardScaler


# class AnomalyAttention:
#     def __init__(self, d_model, dropout_rate=0.1):
#         self.d_model = d_model
#         self.dropout_rate = dropout_rate

#         # Initialize weights with small random values
#         scale = 0.01
#         self.Wq = np.random.randn(d_model, d_model) * scale
#         self.Wk = np.random.randn(d_model, d_model) * scale
#         self.Wv = np.random.randn(d_model, d_model) * scale
#         self.Ws = np.random.randn(d_model, 1) * scale

#     def dropout(self, X):
#         mask = np.random.binomial(1, 1-self.dropout_rate, X.shape)
#         return X * mask / (1-self.dropout_rate)

#     def prior_association(self, sigma):
#         N = len(sigma)
#         p = np.abs(np.arange(N).reshape(-1, 1) - np.arange(N).reshape(1, -1))

#         # Ensure sigma is positive and stable
#         sigma = np.maximum(sigma, 1e-6)

#         # Compute Gaussian in log space for stability
#         log_gaussian = -0.5 * (p / sigma[:, None]) ** 2 - np.log(sigma[:, None]) - 0.5 * np.log(2 * np.pi)
#         log_gaussian = np.clip(log_gaussian, -100, 100)  # Prevent extreme values

#         gaussian = np.exp(log_gaussian)
#         sum_gaussian = np.sum(gaussian, axis=1, keepdims=True)
#         sum_gaussian = np.maximum(sum_gaussian, 1e-10)  # Prevent division by zero

#         prior_ass = gaussian / sum_gaussian
#         return prior_ass

#     def forward(self, x):
#         Q = np.dot(x, self.Wq)
#         K = np.dot(x, self.Wk)
#         V = np.dot(x, self.Wv)
#         sigma = np.maximum(np.dot(x, self.Ws).flatten(), 1e-6)

#         P = self.prior_association(sigma)

#         # Scaled dot-product attention with numerical stability
#         S = np.dot(Q, K.T) / np.sqrt(self.d_model)

#         # Normalize attention scores
#         mean = S.mean(axis=-1, keepdims=True)
#         std = np.maximum(S.std(axis=-1, keepdims=True), 1e-6)
#         S = (S - mean) / std

#         # Softmax with numerical stability
#         S = np.exp(S - S.max(axis=-1, keepdims=True))
#         S = S / np.maximum(S.sum(axis=-1, keepdims=True), 1e-10)

#         # Apply dropout
#         if self.dropout_rate > 0:
#             S = self.dropout(S)

#         Z = np.dot(S, V)
#         return Z, P, S


# class AnomalyTransformer:
#     def __init__(self, d_model, hidden_dim, lambda_=0.01):
#         self.d_model = d_model
#         self.hidden_dim = hidden_dim
#         self.lambda_ = lambda_
#         self.attention = AnomalyAttention(d_model)

#         # Initialize feedforward weights
#         scale = 0.01
#         self.W1 = np.random.randn(d_model, hidden_dim) * scale
#         self.W2 = np.random.randn(hidden_dim, d_model) * scale
#         self.b1 = np.zeros(hidden_dim)
#         self.b2 = np.zeros(d_model)

#     def layer_norm(self, x, eps=1e-6):
#         mean = np.mean(x, axis=-1, keepdims=True)
#         std = np.maximum(np.std(x, axis=-1, keepdims=True), eps)
#         return (x - mean) / std

#     def relu(self, x):
#         return np.maximum(0, x)

#     def forward(self, x):
#         # Attention layer
#         Z, P, S = self.attention.forward(x)
#         x = self.layer_norm(x + Z)

#         # Feedforward network
#         hidden = self.relu(np.dot(x, self.W1) + self.b1)
#         output = np.dot(hidden, self.W2) + self.b2
#         x_hat = self.layer_norm(x + output)

#         return x_hat, [P], [S]

#     def association_discrepancy(self, P_list, S_list):
#         ass_disc = 0
#         for P, S in zip(P_list, S_list):
#             # Add small epsilon to prevent log(0)
#             P = np.maximum(P, 1e-10)
#             S = np.maximum(S, 1e-10)

#             # Symmetric KL divergence
#             kl_div = np.sum(P * np.log(P / S)) + np.sum(S * np.log(S / P))
#             kl_div = np.clip(kl_div, 0, 100)  # Prevent extreme values
#             ass_disc += kl_div

#         return ass_disc / len(P_list)

#     def loss_function(self, x_hat, x, P_list, S_list):
#         reconstruction_loss = np.mean((x_hat - x) ** 2)
#         association_loss = self.association_discrepancy(P_list, S_list)

#         # Scale losses to prevent one term from dominating
#         reconstruction_loss = np.clip(reconstruction_loss, 0, 100)
#         association_loss = np.clip(association_loss, 0, 100)

#         return reconstruction_loss + self.lambda_ * association_loss

#     def anomaly_score(self, x):
#         x_hat, P_list, S_list = self.forward(x)
#         reconstruction_error = np.linalg.norm((x - x_hat) ** 2, axis=1)
#         assoc_dis = self.association_discrepancy(P_list, S_list)
#         ad = np.exp(-assoc_dis) / np.sum(np.exp(-assoc_dis))
#         return ad * reconstruction_error


# def train_batch(model, batch_data, learning_rate=1e-4):
#     """Train model on a single batch using simple gradient descent"""
#     x_hat, P_list, S_list = model.forward(batch_data)
#     loss = model.loss_function(x_hat, batch_data, P_list, S_list)

#     # Backpropagate and update weights
#     grad_output = 2 * (x_hat - batch_data) / batch_data.shape[0]
#     grad_hidden = grad_output @ model.W2.T
#     grad_hidden[hidden <= 0] = 0  # ReLU derivative
#     model.W2 -= learning_rate * np.dot(grad_hidden.T, x_hat) / batch_data.shape[0]
#     model.W1 -= learning_rate * np.dot(batch_data.T, grad_hidden) / batch_data.shape[0]
    
#     return loss


# def main():
#     # Load data from command line
#     if len(sys.argv) < 2:
#         print("Error: No file path provided.")
#         sys.exit(1)

#     input_file = sys.argv[1]

#     # Validate file existence
#     if not os.path.exists(input_file):
#         print(f"Error: The file '{input_file}' does not exist.")
#         sys.exit(1)

#     # Load and process the data
#     df = pd.read_csv(input_file, delimiter=',', on_bad_lines='skip')
#     df = df.apply(pd.to_numeric, errors='coerce')
#     df = df.fillna(df.mean())

#     # Standardize and clip the data
#     scaler = StandardScaler()
#     X = scaler.fit_transform(df.values)
#     X = np.clip(X, -5, 5)  # Clip extreme values

#     # Model parameters
#     d_model = X.shape[1]
#     hidden_dim = 32
#     batch_size = 16
#     epochs = 10

#     # Initialize model
#     model = AnomalyTransformer(d_model, hidden_dim)

#     # Training loop
#     n_batches = len(X) // batch_size
#     for epoch in range(epochs):
#         total_loss = 0

#         # Shuffle data
#         np.random.shuffle(X)

#         for i in range(n_batches):
#             batch = X[i * batch_size:(i + 1) * batch_size]
#             loss = train_batch(model, batch)

#             if not np.isnan(loss) and not np.isinf(loss):
#                 total_loss += loss

#         avg_loss = total_loss / n_batches
#         print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

#     # Calculate anomaly scores
#     anomaly_scores = model.anomaly_score(X)
#     normalized_scores = 0.1 + 0.8 * (
#         (anomaly_scores - anomaly_scores.min()) /
#         (anomaly_scores.max() - anomaly_scores.min())
#     )

#     # Print anomalies
#     for i, score in enumerate(normalized_scores):
#         if score > 0.8:
#             print(f"Anomaly at index {i+2}: score = {score:.4f}")


# if __name__ == "__main__":
#     main()


import json
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load and preprocess data
if len(sys.argv) < 2:
    print("Error: No file path provided.")
    sys.exit(1)

input_file = sys.argv[1]

# Validate file existence
if not os.path.exists(input_file):
    print(f"Error: The file '{input_file}' does not exist.")
    sys.exit(1)

# Load and process the data
df = pd.read_csv(input_file, delimiter=',', on_bad_lines='skip')
df = df.apply(pd.to_numeric, errors='coerce')
df = df.fillna(df.mean())

X = df.values
X_train = torch.FloatTensor(X)

# Normalize data
X_train = (X_train - X_train.mean(dim=0)) / (X_train.std(dim=0) + 1e-8)
X_train = torch.nan_to_num(X_train, nan=0.0, posinf=1.0, neginf=-1.0)

# Define the AnomalyAttention module
class AnomalyAttention(nn.Module):
    def __init__(self, N, d_model):
        super(AnomalyAttention, self).__init__()
        self.N = N
        self.d_model = d_model
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Ws = nn.Linear(d_model, 1, bias=False)

    def forward(self, x):
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)
        sigma = torch.clamp(self.Ws(x), min=1e-3, max=1.0)
        P = self.prior_association(sigma)
        S = (Q @ K.T) / torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        S = torch.softmax(S, dim=-1)
        Z = S @ V
        return Z, P, S

    @staticmethod
    def prior_association(sigma):
        N = sigma.shape[0]
        p = torch.arange(N).unsqueeze(0).repeat(N, 1)
        diff = torch.abs(p - p.T).float()
        gaussian = torch.exp(-0.5 * (diff / sigma).pow(2)) / torch.sqrt(2 * torch.pi * sigma)
        return gaussian / (gaussian.sum(dim=1, keepdim=True) + 1e-8)

# Define the AnomalyTransformer module
class AnomalyTransformer(nn.Module):
    def __init__(self, N, d_model, hidden_dim, lambda_=0.1):
        super(AnomalyTransformer, self).__init__()
        self.N = N
        self.d_model = d_model
        self.lambda_ = lambda_
        self.hidden_dim = hidden_dim
        self.attention_layers = AnomalyAttention(N, d_model)
        self.hidden_layer = nn.Linear(d_model, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, d_model)

    def forward(self, x):
        Z, P, S = self.attention_layers(x)
        hidden = torch.relu(self.hidden_layer(Z))
        x_hat = self.output_layer(hidden)
        return x_hat, P, S

    def anomaly_score(self, x):
        """
        Calculate the anomaly score for each input sample.
        """
        x_hat, P, S = self(x)  # Forward pass
        reconstruction_error = torch.linalg.norm(x - x_hat, dim=1)
        assoc_discrepancy = torch.sum(F.kl_div(torch.log(P + 1e-8), S, reduction='none'), dim=1)
        return reconstruction_error + assoc_discrepancy

# Define the loss functions for minimization and maximization phases
def minimize_phase_loss(x, x_hat, P, S, lambda_):
    reconstruction_loss = torch.linalg.norm(x - x_hat, ord='fro')
    kl_divergence = torch.sum(F.kl_div(torch.log(P + 1e-8), S.detach(), reduction='batchmean'))
    return reconstruction_loss + lambda_ * kl_divergence

def maximize_phase_loss(x, x_hat, P, S, lambda_):
    reconstruction_loss = torch.linalg.norm(x - x_hat, ord='fro')
    kl_divergence = torch.sum(F.kl_div(torch.log(P.detach() + 1e-8), S, reduction='batchmean'))
    return reconstruction_loss + lambda_ * kl_divergence

# Training function with minimization and maximization phases
def train_minimax(model, data, optimizer, lambda_, epochs):
    model.train()
    total_loss = 0.0
    for epoch in range(epochs):
        optimizer.zero_grad()

        # Forward pass
        x_hat, P, S = model(data)

        # Minimization phase
        minimize_loss = minimize_phase_loss(data, x_hat, P, S, lambda_)
        minimize_loss.backward(retain_graph=True)

        # Maximization phase
        maximize_loss = maximize_phase_loss(data, x_hat, P, S, lambda_)
        maximize_loss.backward()

        # Gradient clipping and optimizer step
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += (minimize_loss.item() + maximize_loss.item())
        print(f"Epoch {epoch + 1}/{epochs}, Minimize Loss: {minimize_loss.item():.4f}, Maximize Loss: {maximize_loss.item():.4f}",file=sys.stderr)

    return total_loss / epochs  # Return the average loss over epochs

# Initialize the model and optimizer
N, d_model = X_train.shape
hidden_dim = 64
lambda_ = 0.1
model = AnomalyTransformer(N, d_model, hidden_dim, lambda_)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epochs = 10

# Train the model
average_loss = train_minimax(model, X_train, optimizer, lambda_, epochs)

# Calculate anomaly scores
anomaly_scores = model.anomaly_score(X_train).detach().numpy()
anomaly_scores = np.nan_to_num(anomaly_scores, nan=0.0, posinf=1.0, neginf=-1.0)

# Apply minimax strategy to refine the anomaly detection threshold
minimax_scores = []
for i, score in enumerate(anomaly_scores):
    minimax_scores.append(max(score, min(anomaly_scores)))

# Simulate labels
np.random.seed(42)
true_labels = np.zeros(N, dtype=int)
anomaly_indices = np.random.choice(N, size=int(0.1 * N), replace=False)
true_labels[anomaly_indices] = 1

# Threshold and normalize scores
threshold = np.percentile(minimax_scores, 90)
predictions = (minimax_scores > threshold).astype(int)

# Get the indices of anomalies (where prediction is 1)
anomaly_indices_detected = np.where(predictions == 1)[0].tolist()

# Evaluate metrics
accuracy = accuracy_score(true_labels, predictions)
precision = precision_score(true_labels, predictions, zero_division=0)
recall = recall_score(true_labels, predictions, zero_division=0)
f1 = f1_score(true_labels, predictions, zero_division=0)
roc_auc = roc_auc_score(true_labels, minimax_scores)

# Create the result JSON
result = {
    "anomalies": anomaly_indices_detected,
    "losses": [float(average_loss)] * epochs,
    "threshold": float(threshold),
    "combined_scores": [float(score) for score in minimax_scores]
}
# Print metrics
# print(f"Accuracy: {accuracy * 100:.2f}%")
# print(f"Precision: {precision * 100:.2f}%")
# print(f"Recall: {recall * 100:.2f}%")
# print(f"F1 Score: {f1 * 100:.2f}%")
# print(f"ROC-AUC: {roc_auc:.4f}")
# Print metrics
output_json = json.dumps(result, indent=4)
print(output_json)
