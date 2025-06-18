import csv
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import label_binarize
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Standard amino acids
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INT = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

# Maps for labels
STRUCTURE_TO_INT_SST3 = {'H': 0, 'E': 1, 'C': 2}
STRUCTURE_TO_INT_SST8 = {'H': 0, 'G': 1, 'I': 2, 'E': 3, 'B': 4, 'T': 5, 'S': 6, 'C': 7}


def load_and_preprocess_data(train_path, test_path, structure_type='sst3'):
    
    # load data and filter out non-standard amino acids
    train_df = pd.read_csv(train_path)
    train_df['is_standard'] = train_df['seq'].apply(lambda x: all(aa in AMINO_ACIDS for aa in x))
    train_df = train_df[train_df['is_standard']]
    X_train, y_train = preprocess_data(train_df, structure_type=structure_type)
    
    # load test data and filter out non-standard amino acids
    test_df = pd.read_csv(test_path)
    test_df['is_standard'] = test_df['seq'].apply(lambda x: all(aa in AMINO_ACIDS for aa in x))
    test_df = test_df[test_df['is_standard']]
    X_test, y_test = preprocess_data(test_df, structure_type=structure_type)

    return X_train, y_train, X_test, y_test



def normalize_features(features):
    return (features - np.min(features)) / (np.max(features) - np.min(features))

def one_hot_encode_sequence(sequence):
    one_hot = np.zeros((len(sequence), len(AMINO_ACIDS)), dtype=np.float32)
    for i, aa in enumerate(sequence):
        if aa in AA_TO_INT:
            one_hot[i, AA_TO_INT[aa]] = 1.0
    return one_hot

def encode_labels(labels, structure_map):
    return np.array([structure_map[label] for label in labels])

def pad_sequences(sequences, max_len, padding_value=0.0):
    padded = np.full((len(sequences), max_len, len(AMINO_ACIDS)), padding_value, dtype=np.float32)
    for i, seq in enumerate(sequences):
        padded[i, :len(seq), :] = seq
    return padded

def pad_labels(labels, max_len, padding_value=-1):
    padded = np.full((len(labels), max_len), padding_value, dtype=np.int32)
    for i, label in enumerate(labels):
        padded[i, :len(label)] = label
    return padded

def preprocess_data(df, structure_type='sst3'):
    structure_map = STRUCTURE_TO_INT_SST3 if structure_type == 'sst3' else STRUCTURE_TO_INT_SST8
    sequences = [normalize_features(one_hot_encode_sequence(seq)) for seq in df['seq']]
    labels = [encode_labels(label, structure_map) for label in df[structure_type]]
    max_len = max(len(seq) for seq in df['seq'])
    sequences_padded = pad_sequences(sequences, max_len)
    labels_padded = pad_labels(labels, max_len)
    return sequences_padded, labels_padded

def calculate_metrics(y_true, y_pred, num_classes):
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    sensitivities = cm.diagonal() / cm.sum(axis=1)
    specificities = (cm.sum(axis=0) - cm.diagonal()) / (cm.sum(axis=0) + cm.sum(axis=1) - cm.diagonal())
    return sensitivities, specificities, cm

def save_results_to_csv(results, filename='experiment_results.csv'):
    file_exists = False
    try:
        with open(filename, 'r') as f:
            file_exists = True
    except FileNotFoundError:
        pass
    
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        
        if not file_exists:
             writer.writerow(["algorithm_type", "num_layers", "learning_rate", "batch_size", "hidden_size", "dropout_rate", "train_accuracy", "val_accuracy"])
        
        writer.writerow(results)
        

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout_rate, rnn_type, device='cuda'):
        super(RNNModel, self).__init__()

        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        
        # choose the type of RNN
        if rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
        else:
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
        
        self.fc = nn.Linear(hidden_size, output_size)

        # Dropout layer to prevent overfitting (applied after the RNN layer)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        if isinstance(self.rnn, nn.LSTM):
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
            out, _ = self.rnn(x, (h0, c0))
        else:
            out, _ = self.rnn(x, h0)

        out = self.dropout(out) 
        out = self.fc(out)
        return out


def train_and_evaluate(X_train, y_train, X_test, y_test, input_size, output_size, hidden_size = 512, num_epochs=10, batch_size=16, lr= 0.00406991235625497, num_layers = 3 , dropout_rate = 0.2, rnn_type='LSTM'):
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RNNModel(input_size, hidden_size, output_size, num_layers, dropout_rate, rnn_type, device=device).to(device)  
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    test_losses = []
    test_accuracies = []
    train_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct_train, total_train = 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, output_size), labels.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(outputs, dim=2)
            mask = (labels != -1)
            correct_train += (predicted[mask] == labels[mask]).sum().item()
            total_train += mask.sum().item()

        train_losses.append(total_loss / len(train_loader))
        train_accuracies.append(100 * correct_train / total_train)

        model.eval()
        test_loss, total, correct = 0, 0, 0
        y_true, y_pred  = [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.view(-1, output_size), labels.view(-1))
                test_loss += loss.item()
                _, predicted = torch.max(outputs, dim=2)
                mask = (labels != -1)
                total += mask.sum().item()
                correct += (predicted[mask] == labels[mask]).sum().item()
                y_true.extend(labels[mask].cpu().numpy())
                y_pred.extend(predicted[mask].cpu().numpy())
                
        test_losses.append(test_loss / len(test_loader))
        test_accuracies.append(100 * correct / total)
        
        f1 = f1_score(y_true, y_pred, average='weighted')
        sensitivities, specificities, cm = calculate_metrics(y_true, y_pred, output_size)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}, Test Accuracy: {test_accuracies[-1]:.2f}%")

    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(12, 7))

    plt.subplot(2, 1, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, test_losses, label="Test Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss (Overfitting Analysis)")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(epochs, train_accuracies, label="Train Accuracy")
    plt.plot(epochs, test_accuracies, label="Test Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Training vs Test Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()
    
    print("\nClass-wise Metrics:")
    print(f"{'Class':<5} {'Sensitivity':<12} {'Specificity':<12}")
    for i, (sens, spec) in enumerate(zip(sensitivities, specificities)):
        print(f"{i}       {sens:.4f}        {spec:.4f}")


    print("F1 Score:", f1)
    

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['H', 'E', 'C'])
    disp.plot(cmap="Blues")
    plt.show()
    
    avg_train_accuracy = sum(train_accuracies) / len(train_accuracies)
    avg_val_accuracy = sum(test_accuracies) / len(test_accuracies)
    
    results = [rnn_type, num_layers, lr, batch_size, hidden_size, dropout_rate, avg_train_accuracy, avg_val_accuracy]
    save_results_to_csv(results)
    
    

def random_search(num_experiments, X_train, y_train, X_test, y_test, input_size, output_size):
    
    lr_values = [1e-3, 1e-4, 1e-5, 5e-4, 1e-2] 
    hidden_sizes = [64, 128, 256, 512]  
    dropout_rates = [0.1, 0.3, 0.5, 0.7]  
    num_layers_values = [1, 2, 3, 4]  
    batch_sizes = [16, 32, 64, 128] 
    rnn_types = ['GRU', 'LSTM', 'RNN']  
    
    for _ in range(num_experiments):
    
        lr = random.choice(lr_values)
        hidden_size = random.choice(hidden_sizes)
        dropout_rate = random.choice(dropout_rates)
        num_layers = random.choice(num_layers_values)
        batch_size = random.choice(batch_sizes)
        rnn_type = random.choice(rnn_types)
        
        print(f"\nStarting experiment with parameters: lr={lr}, hidden_size={hidden_size}, dropout_rate={dropout_rate}, num_layers={num_layers}, batch_size={batch_size}, rnn_type={rnn_type}")
        
    
        train_and_evaluate(X_train, y_train, X_test, y_test, input_size,output_size, hidden_size,
                           num_epochs=20, batch_size=batch_size, lr=lr, 
                           num_layers=num_layers, dropout_rate=dropout_rate, rnn_type=rnn_type)

def main():
    train_path = "Dataset\\training_secondary_structure_train.csv"
    test_path = "Dataset\\test_secondary_structure_cb513.csv"

    X_train, y_train, X_test, y_test = load_and_preprocess_data(train_path, test_path)
    print("Shape of test data (X):", X_test.shape)
    print("Shape of test labels (y):", y_test.shape)
    print("Shape of training data (X):", X_train.shape)
    print("Shape of training labels (y):", y_train.shape)

    input_size = len(AMINO_ACIDS)
    output_size = len(STRUCTURE_TO_INT_SST3)
    
    train_and_evaluate(X_train, y_train, X_test, y_test, input_size, output_size)
    
    #random_search(20, X_train, y_train, X_test, y_test, input_size, output_size)

if __name__ == "__main__":
    main()
