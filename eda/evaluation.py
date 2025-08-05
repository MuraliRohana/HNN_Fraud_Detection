import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    
    def __init__(self):
        self.metrics_history = []
        self.predictions_history = []
    
    def evaluate_model(self, model, graph_data, sequence_data, y_true, threshold=0.5):

        model.eval()
        
        with torch.no_grad():
            # Forward pass
            if hasattr(model, 'predict'):
                predictions, probabilities = model.predict(graph_data, sequence_data)
                predictions = predictions.cpu().numpy()
                probabilities = probabilities.cpu().numpy()
            else:
                logits = model(graph_data, sequence_data)
                probabilities = torch.sigmoid(logits).squeeze().cpu().numpy()
                predictions = (probabilities > threshold).astype(int)
        
        # Store predictions
        self.predictions_history.append({
            'y_true': y_true,
            'y_pred': predictions,
            'y_prob': probabilities
        })
        
        return predictions, probabilities
    
    def calculate_metrics(self, y_true, y_pred, y_prob):

        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
        
        # AUC metrics
        if len(np.unique(y_true)) > 1:  # Only if both classes present
            metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
            metrics['auc_pr'] = average_precision_score(y_true, y_prob)
        else:
            metrics['auc_roc'] = 0.0
            metrics['auc_pr'] = 0.0
        
        # Confusion matrix elements
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['true_negatives'] = tn
        metrics['false_positives'] = fp
        metrics['false_negatives'] = fn
        metrics['true_positives'] = tp
        
        # Additional metrics
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
        metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
        metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
        
        # Store metrics
        self.metrics_history.append(metrics)
        
        return metrics
    
    def calculate_roc_curve(self, y_true, y_prob):
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        return fpr, tpr, thresholds
    
    def calculate_pr_curve(self, y_true, y_prob):
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        return precision, recall, thresholds
    
    def plot_confusion_matrix(self, y_true, y_pred, normalize=False):

        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = "Normalized Confusion Matrix"
            text_format = '.2%'
        else:
            title = "Confusion Matrix"
            text_format = 'd'
        
        fig = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=['Legitimate', 'Fraudulent'],
            y=['Legitimate', 'Fraudulent'],
            title=title,
            text_auto=text_format,
            aspect="auto"
        )
        
        return fig
    
    def plot_roc_curve(self, y_true, y_prob):

        fpr, tpr, _ = self.calculate_roc_curve(y_true, y_prob)
        auc_score = roc_auc_score(y_true, y_prob)
        
        fig = go.Figure()
        
        # ROC curve
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {auc_score:.4f})',
            line=dict(color='blue', width=2)
        ))
        
        # Diagonal line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='red', width=1, dash='dash')
        ))
        
        fig.update_layout(
            title='ROC Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            showlegend=True,
            width=600,
            height=500
        )
        
        return fig
    
    def plot_precision_recall_curve(self, y_true, y_prob):

        precision, recall, _ = self.calculate_pr_curve(y_true, y_prob)
        auc_pr = average_precision_score(y_true, y_prob)
        
        fig = go.Figure()
        
        # PR curve
        fig.add_trace(go.Scatter(
            x=recall, y=precision,
            mode='lines',
            name=f'PR Curve (AUC = {auc_pr:.4f})',
            line=dict(color='green', width=2)
        ))
        
        # Baseline (random classifier)
        baseline = np.sum(y_true) / len(y_true)
        fig.add_hline(
            y=baseline,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Baseline (Random): {baseline:.4f}"
        )
        
        fig.update_layout(
            title='Precision-Recall Curve',
            xaxis_title='Recall',
            yaxis_title='Precision',
            showlegend=True,
            width=600,
            height=500
        )
        
        return fig
    
    def plot_threshold_analysis(self, y_true, y_prob):

        thresholds = np.linspace(0, 1, 101)
        precision_scores = []
        recall_scores = []
        f1_scores = []
        
        for threshold in thresholds:
            y_pred = (y_prob > threshold).astype(int)
            
            precision_scores.append(precision_score(y_true, y_pred, zero_division=0))
            recall_scores.append(recall_score(y_true, y_pred, zero_division=0))
            f1_scores.append(f1_score(y_true, y_pred, zero_division=0))
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=thresholds, y=precision_scores,
            mode='lines',
            name='Precision',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=thresholds, y=recall_scores,
            mode='lines',
            name='Recall',
            line=dict(color='red')
        ))
        
        fig.add_trace(go.Scatter(
            x=thresholds, y=f1_scores,
            mode='lines',
            name='F1 Score',
            line=dict(color='green')
        ))
        
        # Find optimal threshold (max F1)
        optimal_threshold = thresholds[np.argmax(f1_scores)]
        fig.add_vline(
            x=optimal_threshold,
            line_dash="dash",
            line_color="black",
            annotation_text=f"Optimal Threshold: {optimal_threshold:.3f}"
        )
        
        fig.update_layout(
            title='Metrics vs Threshold',
            xaxis_title='Threshold',
            yaxis_title='Score',
            showlegend=True,
            width=800,
            height=500
        )
        
        return fig
    
    def plot_feature_importance(self, model, feature_names):

        # This is a placeholder - actual implementation depends on model architecture
        # For neural networks, this might involve gradient-based importance or SHAP values
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            # For neural networks, we can't easily get feature importance
            # This would require additional analysis like SHAP or permutation importance
            return None
        
        # Sort features by importance
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=True)
        
        fig = px.bar(
            feature_importance_df.tail(20),  # Top 20 features
            x='importance',
            y='feature',
            orientation='h',
            title='Top 20 Feature Importances'
        )
        
        return fig
    
    def create_evaluation_report(self, y_true, y_pred, y_prob, feature_names=None):

        report = {}
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_true, y_pred, y_prob)
        report['metrics'] = metrics
        
        # Classification report
        class_report = classification_report(y_true, y_pred, output_dict=True)
        report['classification_report'] = class_report
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        report['confusion_matrix'] = cm
        
        # ROC and PR curves data
        fpr, tpr, roc_thresholds = self.calculate_roc_curve(y_true, y_prob)
        precision, recall, pr_thresholds = self.calculate_pr_curve(y_true, y_prob)
        
        report['roc_curve'] = {'fpr': fpr, 'tpr': tpr, 'thresholds': roc_thresholds}
        report['pr_curve'] = {'precision': precision, 'recall': recall, 'thresholds': pr_thresholds}
        
        # Additional analysis
        report['class_distribution'] = {
            'fraud_rate': np.mean(y_true),
            'total_samples': len(y_true),
            'fraud_samples': np.sum(y_true),
            'legitimate_samples': len(y_true) - np.sum(y_true)
        }
        
        return report
    
    def compare_models(self, model_results):

        comparison_data = []
        
        for model_name, results in model_results.items():
            metrics = results['metrics']
            comparison_data.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1 Score': metrics['f1'],
                'AUC-ROC': metrics['auc_roc'],
                'AUC-PR': metrics['auc_pr']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Create comparison plot
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC', 'AUC-PR']
        
        fig = go.Figure()
        
        for metric in metrics_to_plot:
            fig.add_trace(go.Bar(
                name=metric,
                x=comparison_df['Model'],
                y=comparison_df[metric]
            ))
        
        fig.update_layout(
            title='Model Performance Comparison',
            xaxis_title='Models',
            yaxis_title='Score',
            barmode='group',
            width=1000,
            height=600
        )
        
        return comparison_df, fig
    
    def get_misclassified_samples(self, X, y_true, y_pred, y_prob, top_n=10):

        # Find misclassified samples
        misclassified_mask = (y_true != y_pred)
        
        if not np.any(misclassified_mask):
            return pd.DataFrame()  # No misclassified samples
        
        misclassified_indices = np.where(misclassified_mask)[0]
        
        # Calculate confidence (distance from 0.5)
        confidence = np.abs(y_prob - 0.5)
        
        # Get most confident misclassifications
        misclassified_confidence = confidence[misclassified_mask]
        top_indices = misclassified_indices[np.argsort(misclassified_confidence)[-top_n:]]
        
        # Create DataFrame
        misclassified_df = pd.DataFrame({
            'Index': top_indices,
            'True_Label': y_true[top_indices],
            'Predicted_Label': y_pred[top_indices],
            'Probability': y_prob[top_indices],
            'Confidence': confidence[top_indices]
        })
        
        return misclassified_df.sort_values('Confidence', ascending=False)
