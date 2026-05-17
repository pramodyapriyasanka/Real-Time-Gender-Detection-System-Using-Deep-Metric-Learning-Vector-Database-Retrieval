import pickle
from pathlib import Path
import numpy as np
import faiss
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class GenderDataset(Dataset):
    """Custom Dataset for gender classification"""
    def __init__(self, embeddings, labels):
        self.embeddings = torch.FloatTensor(embeddings)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


class MLPGenderClassifier(nn.Module):
    """MLP Neural Network for Gender Classification from 512D embeddings"""
    def __init__(self, input_dim=512, hidden_dims=[256, 128, 64], dropout=0.3):
        super(MLPGenderClassifier, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 2))  # 2 classes: Female, Male
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class MLPTrainer:
    """Trainer class for MLP model"""
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=3
        )
        
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for embeddings, labels in train_loader:
            embeddings, labels = embeddings.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(embeddings)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        return total_loss / len(train_loader), 100 * correct / total
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for embeddings, labels in val_loader:
                embeddings, labels = embeddings.to(self.device), labels.to(self.device)
                
                outputs = self.model(embeddings)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = 100 * correct / total
        return total_loss / len(val_loader), accuracy, all_preds, all_labels
    
    def train(self, train_loader, val_loader, epochs=50, save_path='mlp_gender_model.pth'):
        """Full training loop"""
        best_val_acc = 0
        patience_counter = 0
        max_patience = 10
        
        print("Starting training...")
        print(f"Device: {self.device}")
        print(f"Total epochs: {epochs}\n")
        
        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc, val_preds, val_labels = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_acc)
            
            # Print progress
            print(f"Epoch [{epoch+1}/{epochs}]")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                }, save_path)
                print(f"  ✓ New best model saved! (Val Acc: {val_acc:.2f}%)")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= max_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
            
            print()
        
        print(f"\nTraining completed!")
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
        return best_val_acc


def load_embeddings_from_faiss(faiss_index_path, metadata_path):
    """Load embeddings and labels from FAISS index and metadata"""
    print("Loading FAISS index...")
    index = faiss.read_index(faiss_index_path)
    
    print("Loading metadata...")
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    print(f"Extracting {index.ntotal} embeddings from FAISS index...")
    embeddings = np.zeros((index.ntotal, 512), dtype=np.float32)
    
    # Extract all embeddings from FAISS
    for i in range(index.ntotal):
        embedding = index.reconstruct(int(i))
        embeddings[i] = embedding
    
    # Extract labels
    labels = []
    for meta in metadata:
        label = 0 if meta['class'] == 'Female' else 1  # Female=0, Male=1
        labels.append(label)
    
    labels = np.array(labels)
    
    print(f"Loaded {len(embeddings)} embeddings with shape {embeddings.shape}")
    print(f"Label distribution: Female={np.sum(labels==0)}, Male={np.sum(labels==1)}")
    
    return embeddings, labels


def main():
    # Paths
    base_dir = Path(r"C:\Users\menuk\Desktop\AGE")
    faiss_index = base_dir / "embeddings" / "face_embeddings.index"
    metadata = base_dir / "embeddings" / "metadata.pkl"
    model_save_path = base_dir / "models" / "mlp_gender_model.pth"
    
    # Create models directory
    model_save_path.parent.mkdir(exist_ok=True)
    
    # Check GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    else:
        print("Warning: Training on CPU will be slower\n")
    
    # Load data
    print("="*60)
    print("STEP 1: Loading Data")
    print("="*60 + "\n")
    embeddings, labels = load_embeddings_from_faiss(str(faiss_index), str(metadata))
    
    # Split data
    print("\n" + "="*60)
    print("STEP 2: Splitting Data")
    print("="*60 + "\n")
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=0.2, random_state=42, stratify=labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Create datasets and loaders
    train_dataset = GenderDataset(X_train, y_train)
    val_dataset = GenderDataset(X_val, y_val)
    test_dataset = GenderDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)
    
    # Create model
    print("\n" + "="*60)
    print("STEP 3: Creating Model")
    print("="*60 + "\n")
    model = MLPGenderClassifier(
        input_dim=512,
        hidden_dims=[256, 128, 64],
        dropout=0.3
    )
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    print("\n" + "="*60)
    print("STEP 4: Training Model")
    print("="*60 + "\n")
    trainer = MLPTrainer(model, device=device)
    best_val_acc = trainer.train(
        train_loader, 
        val_loader, 
        epochs=50,
        save_path=str(model_save_path)
    )
    
    # Load best model for testing
    print("\n" + "="*60)
    print("STEP 5: Evaluating on Test Set")
    print("="*60 + "\n")
    checkpoint = torch.load(str(model_save_path))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Test evaluation
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for embeddings_batch, labels_batch in test_loader:
            embeddings_batch = embeddings_batch.to(device)
            outputs = model(embeddings_batch)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels_batch.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
    test_acc = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {test_acc*100:.2f}%\n")
    
    print("Classification Report:")
    print(classification_report(
        all_labels, 
        all_preds, 
        target_names=['Female', 'Male'],
        digits=4
    ))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(f"                Predicted")
    print(f"              Female    Male")
    print(f"Actual Female {cm[0][0]:6d}  {cm[0][1]:6d}")
    print(f"       Male   {cm[1][0]:6d}  {cm[1][1]:6d}")
    
    print("\n" + "="*60)
    print(f"Model saved to: {model_save_path}")
    print("="*60)


if __name__ == '__main__':
    main()
