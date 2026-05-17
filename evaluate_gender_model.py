"""
Gender Detection Model Evaluation Script

This script evaluates gender detection systems by computing comprehensive metrics:
- Overall accuracy
- Precision, recall, F1-score per class
- Confusion matrix with heatmap visualization
- ROC curve and AUC score
- Optional: Top-k accuracy for FAISS-based systems

Usage:
    python evaluate_gender_model.py
"""

import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    roc_auc_score
)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    import faiss
except ImportError as e:
    print(f"Warning: Some imports failed: {e}")
    print("Continuing with basic functionality...")


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


def compute_metrics(y_true, y_pred, class_names=['Female', 'Male']):
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: True labels (array-like)
        y_pred: Predicted labels (array-like)
        class_names: Names of the classes
    
    Returns:
        dict: Dictionary containing all computed metrics
    """
    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Precision, recall, F1-score per class
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=[0, 1]
    )
    
    # Macro and weighted averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro'
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Build metrics dictionary
    metrics = {
        'accuracy': accuracy,
        'precision_female': precision[0],
        'precision_male': precision[1],
        'recall_female': recall[0],
        'recall_male': recall[1],
        'f1_female': f1[0],
        'f1_male': f1[1],
        'support_female': support[0],
        'support_male': support[1],
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'confusion_matrix': cm,
        'class_names': class_names
    }
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, class_names=['Female', 'Male'], 
                          save_path='confusion_matrix.png', title='Confusion Matrix'):
    """
    Plot and save confusion matrix as a heatmap.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of the classes
        save_path: Path to save the plot
        title: Title of the plot
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot heatmap with annotations
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'},
        square=True,
        linewidths=1,
        linecolor='gray'
    )
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to: {save_path}")
    
    # Show plot
    plt.show()
    plt.close()


def plot_roc_curve(y_true, y_probs, save_path='roc_curve.png'):
    """
    Plot ROC curve and compute AUC score.
    
    Args:
        y_true: True labels (binary: 0 or 1)
        y_probs: Predicted probabilities for positive class
        save_path: Path to save the plot
    
    Returns:
        float: AUC score
    """
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title('Receiver Operating Characteristic (ROC) Curve', 
              fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"ROC curve saved to: {save_path}")
    
    plt.show()
    plt.close()
    
    return roc_auc


def compute_topk_accuracy(embeddings, labels, index, metadata, k=5):
    """
    Compute top-k accuracy for FAISS-based k-NN classification.
    
    Args:
        embeddings: Test embeddings (numpy array)
        labels: True labels
        index: FAISS index
        metadata: Metadata containing class labels
        k: Number of neighbors to consider
    
    Returns:
        float: Top-k accuracy
    """
    correct = 0
    total = len(embeddings)
    
    for i, embedding in enumerate(embeddings):
        # Search FAISS
        distances, indices = index.search(embedding.reshape(1, -1), k)
        
        # Get neighbor labels
        neighbor_labels = []
        for idx in indices[0]:
            if idx < len(metadata):
                gender = metadata[idx]['class']
                neighbor_labels.append(0 if gender == 'Female' else 1)
        
        # Check if true label is in top-k
        true_label = labels[i]
        if true_label in neighbor_labels:
            correct += 1
    
    return correct / total


def save_metrics_report(metrics, model_name='Model', save_path='metrics_report.csv'):
    """
    Save metrics to CSV and text files.
    
    Args:
        metrics: Dictionary of metrics
        model_name: Name of the model being evaluated
        save_path: Path to save CSV file
    """
    # Create DataFrame for CSV
    data = {
        'Model': [model_name],
        'Accuracy': [metrics['accuracy']],
        'Precision_Female': [metrics['precision_female']],
        'Precision_Male': [metrics['precision_male']],
        'Recall_Female': [metrics['recall_female']],
        'Recall_Male': [metrics['recall_male']],
        'F1_Female': [metrics['f1_female']],
        'F1_Male': [metrics['f1_male']],
        'Support_Female': [metrics['support_female']],
        'Support_Male': [metrics['support_male']],
        'Precision_Macro': [metrics['precision_macro']],
        'Recall_Macro': [metrics['recall_macro']],
        'F1_Macro': [metrics['f1_macro']],
        'Precision_Weighted': [metrics['precision_weighted']],
        'Recall_Weighted': [metrics['recall_weighted']],
        'F1_Weighted': [metrics['f1_weighted']]
    }
    
    if 'auc_score' in metrics:
        data['AUC'] = [metrics['auc_score']]
    
    if 'topk_accuracy' in metrics:
        data['Top-K_Accuracy'] = [metrics['topk_accuracy']]
    
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)
    print(f"\nMetrics saved to: {save_path}")
    
    # Also save detailed text report
    txt_path = save_path.replace('.csv', '.txt')
    with open(txt_path, 'w') as f:
        f.write(f"{'='*60}\n")
        f.write(f"Gender Detection Model Evaluation Report\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*60}\n\n")
        
        f.write(f"Overall Accuracy: {metrics['accuracy']*100:.2f}%\n\n")
        
        f.write(f"Per-Class Metrics:\n")
        f.write(f"{'-'*60}\n")
        f.write(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}\n")
        f.write(f"{'-'*60}\n")
        f.write(f"{'Female':<15} {metrics['precision_female']:<12.4f} {metrics['recall_female']:<12.4f} "
                f"{metrics['f1_female']:<12.4f} {int(metrics['support_female']):<10}\n")
        f.write(f"{'Male':<15} {metrics['precision_male']:<12.4f} {metrics['recall_male']:<12.4f} "
                f"{metrics['f1_male']:<12.4f} {int(metrics['support_male']):<10}\n")
        f.write(f"{'-'*60}\n\n")
        
        f.write(f"Averaged Metrics:\n")
        f.write(f"{'-'*60}\n")
        f.write(f"Macro Avg:    Precision={metrics['precision_macro']:.4f}, "
                f"Recall={metrics['recall_macro']:.4f}, F1={metrics['f1_macro']:.4f}\n")
        f.write(f"Weighted Avg: Precision={metrics['precision_weighted']:.4f}, "
                f"Recall={metrics['recall_weighted']:.4f}, F1={metrics['f1_weighted']:.4f}\n")
        f.write(f"{'-'*60}\n\n")
        
        if 'auc_score' in metrics:
            f.write(f"AUC Score: {metrics['auc_score']:.4f}\n\n")
        
        if 'topk_accuracy' in metrics:
            f.write(f"Top-K Accuracy (k={metrics.get('k', 5)}): {metrics['topk_accuracy']*100:.2f}%\n\n")
        
        f.write(f"Confusion Matrix:\n")
        f.write(f"{'-'*60}\n")
        cm = metrics['confusion_matrix']
        f.write(f"                Predicted\n")
        f.write(f"              Female    Male\n")
        f.write(f"Actual Female {cm[0][0]:6d}  {cm[0][1]:6d}\n")
        f.write(f"       Male   {cm[1][0]:6d}  {cm[1][1]:6d}\n")
        f.write(f"{'-'*60}\n")
    
    print(f"Detailed report saved to: {txt_path}")


def evaluate_mlp_model(model_path, embeddings, labels, device='cuda'):
    """
    Evaluate MLP model on test data.
    
    Args:
        model_path: Path to saved MLP model
        embeddings: Test embeddings
        labels: True labels
        device: Device to run inference on
    
    Returns:
        tuple: (y_pred, y_probs)
    """
    print(f"\nLoading MLP model from {model_path}...")
    
    # Load model
    model = MLPGenderClassifier()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded (Val Acc: {checkpoint.get('val_acc', 0):.2f}%)")
    
    # Create DataLoader
    dataset = TensorDataset(torch.FloatTensor(embeddings), torch.LongTensor(labels))
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    # Predict
    all_preds = []
    all_probs = []
    
    print("Running predictions...")
    with torch.no_grad():
        for batch_embeddings, _ in loader:
            batch_embeddings = batch_embeddings.to(device)
            outputs = model(batch_embeddings)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)
    
    return y_pred, y_probs


def evaluate_faiss_model(index_path, metadata_path, embeddings, labels, k=5):
    """
    Evaluate FAISS k-NN model on test data.
    
    Args:
        index_path: Path to FAISS index
        metadata_path: Path to metadata file
        embeddings: Test embeddings
        labels: True labels
        k: Number of neighbors
    
    Returns:
        tuple: (y_pred, y_probs)
    """
    print(f"\nLoading FAISS index from {index_path}...")
    
    # Load FAISS index
    index = faiss.read_index(index_path)
    
    # Load metadata
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    print(f"FAISS index loaded ({index.ntotal} embeddings)")
    
    # Predict
    all_preds = []
    all_probs = []
    
    print("Running predictions...")
    for embedding in embeddings:
        # Search FAISS
        distances, indices = index.search(embedding.reshape(1, -1), k)
        
        # Count votes
        gender_votes = {'Female': 0, 'Male': 0}
        for idx in indices[0]:
            if idx < len(metadata):
                gender = metadata[idx]['class']
                gender_votes[gender] += 1
        
        # Calculate probabilities
        female_prob = gender_votes['Female'] / k
        male_prob = gender_votes['Male'] / k
        
        # Predict
        if female_prob > male_prob:
            pred = 0  # Female
        else:
            pred = 1  # Male
        
        all_preds.append(pred)
        all_probs.append([female_prob, male_prob])
    
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)
    
    return y_pred, y_probs


def main():
    """Main evaluation pipeline"""
    
    # Paths
    base_dir = Path(r"C:\Users\menuk\Desktop\AGE")
    faiss_index = base_dir / "embeddings" / "face_embeddings.index"
    metadata = base_dir / "embeddings" / "metadata.pkl"
    mlp_model = base_dir / "models" / "mlp_gender_model.pth"
    output_dir = base_dir / "evaluation_results"
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Device selection
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    
    print("="*60)
    print("Gender Detection Model Evaluation")
    print("="*60 + "\n")
    
    # Load test data
    print("Loading test data...")
    
    # Load FAISS index to extract embeddings
    index = faiss.read_index(str(faiss_index))
    with open(metadata, 'rb') as f:
        meta = pickle.load(f)
    
    # Extract all embeddings
    print(f"Extracting {index.ntotal} embeddings...")
    all_embeddings = np.zeros((index.ntotal, 512), dtype=np.float32)
    for i in range(index.ntotal):
        all_embeddings[i] = index.reconstruct(int(i))
    
    # Extract labels
    all_labels = np.array([0 if m['class'] == 'Female' else 1 for m in meta])
    
    # Split data (use same split as training)
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(
        all_embeddings, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )
    
    print(f"Test set: {len(X_test)} samples")
    print(f"Female: {np.sum(y_test==0)}, Male: {np.sum(y_test==1)}\n")
    
    # Evaluate MLP Model
    if mlp_model.exists():
        print("="*60)
        print("Evaluating MLP Model")
        print("="*60)
        
        mlp_pred, mlp_probs = evaluate_mlp_model(str(mlp_model), X_test, y_test, device)
        
        # Compute metrics
        mlp_metrics = compute_metrics(y_test, mlp_pred)
        
        # Print results
        print(f"\nMLP Model Accuracy: {mlp_metrics['accuracy']*100:.2f}%")
        
        # Plot confusion matrix
        plot_confusion_matrix(
            y_test, mlp_pred,
            save_path=str(output_dir / 'mlp_confusion_matrix.png'),
            title='MLP Model - Confusion Matrix'
        )
        
        # Plot ROC curve (for Male class)
        mlp_auc = plot_roc_curve(
            y_test, mlp_probs[:, 1],
            save_path=str(output_dir / 'mlp_roc_curve.png')
        )
        mlp_metrics['auc_score'] = mlp_auc
        
        # Save metrics
        save_metrics_report(
            mlp_metrics, 
            model_name='MLP Neural Network',
            save_path=str(output_dir / 'mlp_metrics_report.csv')
        )
    
    # Evaluate FAISS Model
    print("\n" + "="*60)
    print("Evaluating FAISS k-NN Model")
    print("="*60)
    
    faiss_pred, faiss_probs = evaluate_faiss_model(
        str(faiss_index), str(metadata), X_test, y_test, k=5
    )
    
    # Compute metrics
    faiss_metrics = compute_metrics(y_test, faiss_pred)
    
    # Print results
    print(f"\nFAISS Model Accuracy: {faiss_metrics['accuracy']*100:.2f}%")
    
    # Plot confusion matrix
    plot_confusion_matrix(
        y_test, faiss_pred,
        save_path=str(output_dir / 'faiss_confusion_matrix.png'),
        title='FAISS k-NN Model - Confusion Matrix'
    )
    
    # Plot ROC curve
    faiss_auc = plot_roc_curve(
        y_test, faiss_probs[:, 1],
        save_path=str(output_dir / 'faiss_roc_curve.png')
    )
    faiss_metrics['auc_score'] = faiss_auc
    
    # Compute top-k accuracy
    topk_acc = compute_topk_accuracy(X_test, y_test, index, meta, k=5)
    faiss_metrics['topk_accuracy'] = topk_acc
    faiss_metrics['k'] = 5
    print(f"Top-5 Accuracy: {topk_acc*100:.2f}%")
    
    # Save metrics
    save_metrics_report(
        faiss_metrics,
        model_name='FAISS k-NN (k=5)',
        save_path=str(output_dir / 'faiss_metrics_report.csv')
    )
    
    # Compare models
    if mlp_model.exists():
        print("\n" + "="*60)
        print("Model Comparison")
        print("="*60)
        
        comparison = pd.DataFrame({
            'Model': ['MLP Neural Network', 'FAISS k-NN (k=5)'],
            'Accuracy': [mlp_metrics['accuracy'], faiss_metrics['accuracy']],
            'F1_Female': [mlp_metrics['f1_female'], faiss_metrics['f1_female']],
            'F1_Male': [mlp_metrics['f1_male'], faiss_metrics['f1_male']],
            'F1_Macro': [mlp_metrics['f1_macro'], faiss_metrics['f1_macro']],
            'AUC': [mlp_metrics.get('auc_score', 0), faiss_metrics.get('auc_score', 0)]
        })
        
        print("\n" + comparison.to_string(index=False))
        comparison.to_csv(output_dir / 'model_comparison.csv', index=False)
        print(f"\nComparison saved to: {output_dir / 'model_comparison.csv'}")
    
    print("\n" + "="*60)
    print("Evaluation Complete!")
    print(f"All results saved to: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
