import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class Visualizer:
    """Visualization utilities for fraud detection system"""
    
    def __init__(self):
        self.color_palette = px.colors.qualitative.Set3
        
    def plot_data_distribution(self, data):
        """
        Plot comprehensive data distribution analysis
        
        Args:
            data: Transaction dataframe
        
        Returns:
            Dictionary of plotly figures
        """
        figures = {}
        
        # 1. Fraud distribution pie chart
        fraud_counts = data['Fraud_Label'].value_counts()
        fig_pie = px.pie(
            values=fraud_counts.values,
            names=['Legitimate', 'Fraudulent'],
            title="Transaction Distribution by Fraud Label",
            color_discrete_sequence=['lightblue', 'red']
        )
        figures['fraud_distribution'] = fig_pie
        
        # 2. Transaction amount distribution
        fig_amount = go.Figure()
        
        # Legitimate transactions
        legit_amounts = data[data['Fraud_Label'] == 0]['Transaction_Amount']
        fig_amount.add_trace(go.Histogram(
            x=legit_amounts,
            name='Legitimate',
            opacity=0.7,
            nbinsx=50,
            marker_color='lightblue'
        ))
        
        # Fraudulent transactions
        fraud_amounts = data[data['Fraud_Label'] == 1]['Transaction_Amount']
        fig_amount.add_trace(go.Histogram(
            x=fraud_amounts,
            name='Fraudulent',
            opacity=0.7,
            nbinsx=50,
            marker_color='red'
        ))
        
        fig_amount.update_layout(
            title="Transaction Amount Distribution by Fraud Label",
            xaxis_title="Transaction Amount",
            yaxis_title="Count",
            barmode='overlay'
        )
        figures['amount_distribution'] = fig_amount
        
        # 3. Fraud rate by categorical features
        categorical_features = ['Transaction_Type', 'Device_Type', 'Location', 
                              'Merchant_Category', 'Card_Type', 'Authentication_Method']
        
        for feature in categorical_features:
            if feature in data.columns:
                fraud_rate_data = data.groupby(feature)['Fraud_Label'].agg(['count', 'sum']).reset_index()
                fraud_rate_data['fraud_rate'] = fraud_rate_data['sum'] / fraud_rate_data['count'] * 100
                
                fig = px.bar(
                    fraud_rate_data,
                    x=feature,
                    y='fraud_rate',
                    title=f'Fraud Rate by {feature}',
                    labels={'fraud_rate': 'Fraud Rate (%)'}
                )
                fig.update_layout(xaxis_tickangle=-45)
                figures[f'fraud_rate_{feature.lower()}'] = fig
        
        # 4. Temporal analysis
        if 'Timestamp' in data.columns:
            data_temp = data.copy()
            data_temp['Timestamp'] = pd.to_datetime(data_temp['Timestamp'])
            data_temp['Hour'] = data_temp['Timestamp'].dt.hour
            data_temp['DayOfWeek'] = data_temp['Timestamp'].dt.day_name()
            data_temp['Month'] = data_temp['Timestamp'].dt.month_name()
            
            # Hourly fraud rate
            hourly_fraud = data_temp.groupby('Hour')['Fraud_Label'].agg(['count', 'sum']).reset_index()
            hourly_fraud['fraud_rate'] = hourly_fraud['sum'] / hourly_fraud['count'] * 100
            
            fig_hourly = px.line(
                hourly_fraud,
                x='Hour',
                y='fraud_rate',
                title='Fraud Rate by Hour of Day',
                labels={'fraud_rate': 'Fraud Rate (%)'}
            )
            figures['hourly_fraud_rate'] = fig_hourly
            
            # Weekly fraud rate
            weekly_fraud = data_temp.groupby('DayOfWeek')['Fraud_Label'].agg(['count', 'sum']).reset_index()
            weekly_fraud['fraud_rate'] = weekly_fraud['sum'] / weekly_fraud['count'] * 100
            
            # Reorder days
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            weekly_fraud['DayOfWeek'] = pd.Categorical(weekly_fraud['DayOfWeek'], categories=day_order, ordered=True)
            weekly_fraud = weekly_fraud.sort_values('DayOfWeek')
            
            fig_weekly = px.bar(
                weekly_fraud,
                x='DayOfWeek',
                y='fraud_rate',
                title='Fraud Rate by Day of Week',
                labels={'fraud_rate': 'Fraud Rate (%)'}
            )
            figures['weekly_fraud_rate'] = fig_weekly
        
        return figures
    
    def plot_feature_correlations(self, data, target_col='Fraud_Label'):
        """
        Plot feature correlation heatmap
        
        Args:
            data: DataFrame with features
            target_col: Target column name
        
        Returns:
            Plotly figure
        """
        # Select numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Calculate correlation matrix
        corr_matrix = data[numeric_cols].corr()
        
        # Create heatmap
        fig = px.imshow(
            corr_matrix,
            labels=dict(color="Correlation"),
            title="Feature Correlation Matrix",
            aspect="auto",
            color_continuous_scale='RdBu_r'
        )
        
        fig.update_layout(
            width=800,
            height=600
        )
        
        return fig
    
    def plot_feature_importance_with_target(self, data, target_col='Fraud_Label', top_n=15):
        """
        Plot correlation of features with target variable
        
        Args:
            data: DataFrame
            target_col: Target column
            top_n: Number of top features to show
        
        Returns:
            Plotly figure
        """
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        
        # Calculate correlations with target
        correlations = data[numeric_cols + [target_col]].corr()[target_col].drop(target_col)
        correlations = correlations.abs().sort_values(ascending=True).tail(top_n)
        
        # Create horizontal bar plot
        fig = px.bar(
            x=correlations.values,
            y=correlations.index,
            orientation='h',
            title=f'Top {top_n} Features Correlated with {target_col}',
            labels={'x': 'Absolute Correlation', 'y': 'Features'}
        )
        
        return fig
    
    def plot_graph_structure(self, graph_data, max_nodes=100):
        """
        Visualize graph structure
        
        Args:
            graph_data: PyTorch Geometric Data object
            max_nodes: Maximum number of nodes to visualize
        
        Returns:
            Plotly figure
        """
        # Convert to NetworkX
        edge_list = graph_data.edge_index.t().numpy()
        
        # Limit nodes for visualization
        if graph_data.x.size(0) > max_nodes:
            # Sample nodes
            node_indices = np.random.choice(graph_data.x.size(0), max_nodes, replace=False)
            
            # Filter edges to only include sampled nodes
            edge_mask = np.isin(edge_list[:, 0], node_indices) & np.isin(edge_list[:, 1], node_indices)
            filtered_edges = edge_list[edge_mask]
            
            # Remap node indices
            node_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(node_indices)}
            filtered_edges = np.array([[node_mapping[edge[0]], node_mapping[edge[1]]] 
                                     for edge in filtered_edges if edge[0] in node_mapping and edge[1] in node_mapping])
        else:
            filtered_edges = edge_list
            node_indices = np.arange(graph_data.x.size(0))
        
        # Create NetworkX graph
        G = nx.Graph()
        G.add_nodes_from(range(len(node_indices)))
        if len(filtered_edges) > 0:
            G.add_edges_from(filtered_edges)
        
        # Calculate layout
        if len(G.nodes()) > 0:
            pos = nx.spring_layout(G, k=1, iterations=50)
        else:
            pos = {}
        
        # Create edge traces
        edge_x = []
        edge_y = []
        for edge in G.edges():
            if edge[0] in pos and edge[1] in pos:
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Create node traces
        node_x = []
        node_y = []
        node_text = []
        for node in G.nodes():
            if node in pos:
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(f'Node {node}<br>Degree: {G.degree(node)}')
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                showscale=True,
                colorscale='YlOrRd',
                reversescale=True,
                color=[],
                size=10,
                colorbar=dict(
                    thickness=15,
                    xanchor="left",
                    title="Node Degree"
                ),
                line=dict(width=2)
            )
        )
        
        # Color nodes by degree
        node_degrees = [G.degree(node) for node in G.nodes() if node in pos]
        node_trace.marker.color = node_degrees
        
        # Create figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=f'Graph Structure (Sample of {len(node_indices)} nodes)',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                annotations=[
                    dict(
                        text="Graph visualization with node degree coloring",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002,
                        xanchor='left', yanchor='bottom',
                        font=dict(size=12)
                    )
                ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
        )
        
        return fig
    
    def plot_embedding_visualization(self, embeddings, labels, method='tsne'):
        """
        Visualize high-dimensional embeddings in 2D
        
        Args:
            embeddings: High-dimensional embeddings
            labels: Labels for coloring
            method: Dimensionality reduction method ('tsne' or 'pca')
        
        Returns:
            Plotly figure
        """
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()
        
        # Apply dimensionality reduction
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
        else:  # PCA
            reducer = PCA(n_components=2, random_state=42)
        
        embedded = reducer.fit_transform(embeddings)
        
        # Create scatter plot
        fig = px.scatter(
            x=embedded[:, 0],
            y=embedded[:, 1],
            color=labels.astype(str),
            title=f'Embedding Visualization ({method.upper()})',
            labels={'color': 'Fraud Label'},
            color_discrete_map={'0': 'lightblue', '1': 'red'}
        )
        
        fig.update_layout(
            xaxis_title=f'{method.upper()} Component 1',
            yaxis_title=f'{method.upper()} Component 2'
        )
        
        return fig
    
    def plot_training_history(self, history):
        """
        Plot training history
        
        Args:
            history: Dictionary containing training metrics
        
        Returns:
            Dictionary of plotly figures
        """
        figures = {}
        
        # Training loss
        if 'train_loss' in history:
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(
                x=list(range(len(history['train_loss']))),
                y=history['train_loss'],
                mode='lines',
                name='Training Loss',
                line=dict(color='blue')
            ))
            
            if 'val_loss' in history:
                fig_loss.add_trace(go.Scatter(
                    x=list(range(len(history['val_loss']))),
                    y=history['val_loss'],
                    mode='lines',
                    name='Validation Loss',
                    line=dict(color='red')
                ))
            
            fig_loss.update_layout(
                title='Training and Validation Loss',
                xaxis_title='Epoch',
                yaxis_title='Loss'
            )
            figures['loss'] = fig_loss
        
        # Validation metrics
        metrics = ['val_f1', 'val_precision', 'val_recall', 'val_accuracy']
        available_metrics = [m for m in metrics if m in history]
        
        if available_metrics:
            fig_metrics = go.Figure()
            colors = ['green', 'blue', 'orange', 'purple']
            
            for i, metric in enumerate(available_metrics):
                fig_metrics.add_trace(go.Scatter(
                    x=list(range(len(history[metric]))),
                    y=history[metric],
                    mode='lines',
                    name=metric.replace('val_', '').title(),
                    line=dict(color=colors[i % len(colors)])
                ))
            
            fig_metrics.update_layout(
                title='Validation Metrics',
                xaxis_title='Epoch',
                yaxis_title='Score'
            )
            figures['metrics'] = fig_metrics
        
        return figures
    
    def plot_prediction_distribution(self, y_true, y_prob):
        """
        Plot distribution of prediction probabilities
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
        
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Legitimate transactions
        legit_probs = y_prob[y_true == 0]
        fig.add_trace(go.Histogram(
            x=legit_probs,
            name='Legitimate',
            opacity=0.7,
            nbinsx=50,
            marker_color='lightblue'
        ))
        
        # Fraudulent transactions
        fraud_probs = y_prob[y_true == 1]
        fig.add_trace(go.Histogram(
            x=fraud_probs,
            name='Fraudulent',
            opacity=0.7,
            nbinsx=50,
            marker_color='red'
        ))
        
        # Add threshold line
        fig.add_vline(
            x=0.5,
            line_dash="dash",
            line_color="black",
            annotation_text="Classification Threshold"
        )
        
        fig.update_layout(
            title='Distribution of Prediction Probabilities',
            xaxis_title='Fraud Probability',
            yaxis_title='Count',
            barmode='overlay'
        )
        
        return fig
    
    def plot_risk_analysis(self, data):
        """
        Plot risk analysis charts
        
        Args:
            data: Transaction data
        
        Returns:
            Dictionary of plotly figures
        """
        figures = {}
        
        # Risk score distribution by fraud label
        fig_risk = go.Figure()
        
        legit_risk = data[data['Fraud_Label'] == 0]['Risk_Score']
        fraud_risk = data[data['Fraud_Label'] == 1]['Risk_Score']
        
        fig_risk.add_trace(go.Histogram(
            x=legit_risk,
            name='Legitimate',
            opacity=0.7,
            nbinsx=30,
            marker_color='lightblue'
        ))
        
        fig_risk.add_trace(go.Histogram(
            x=fraud_risk,
            name='Fraudulent',
            opacity=0.7,
            nbinsx=30,
            marker_color='red'
        ))
        
        fig_risk.update_layout(
            title='Risk Score Distribution by Fraud Label',
            xaxis_title='Risk Score',
            yaxis_title='Count',
            barmode='overlay'
        )
        figures['risk_distribution'] = fig_risk
        
        # Transaction amount vs risk score
        fig_scatter = px.scatter(
            data,
            x='Risk_Score',
            y='Transaction_Amount',
            color='Fraud_Label',
            title='Transaction Amount vs Risk Score',
            color_discrete_map={0: 'lightblue', 1: 'red'},
            labels={'Fraud_Label': 'Fraud Label'}
        )
        figures['amount_vs_risk'] = fig_scatter
        
        # Risk score by transaction type
        if 'Transaction_Type' in data.columns:
            risk_by_type = data.groupby(['Transaction_Type', 'Fraud_Label'])['Risk_Score'].mean().reset_index()
            
            fig_type = px.bar(
                risk_by_type,
                x='Transaction_Type',
                y='Risk_Score',
                color='Fraud_Label',
                title='Average Risk Score by Transaction Type',
                color_discrete_map={0: 'lightblue', 1: 'red'},
                barmode='group'
            )
            figures['risk_by_type'] = fig_type
        
        return figures
    
    def create_dashboard_summary(self, data, metrics=None):
        """
        Create summary statistics for dashboard
        
        Args:
            data: Transaction data
            metrics: Model performance metrics
        
        Returns:
            Dictionary of summary statistics
        """
        summary = {}
        
        # Dataset summary
        summary['dataset'] = {
            'total_transactions': len(data),
            'fraud_transactions': data['Fraud_Label'].sum(),
            'fraud_rate': data['Fraud_Label'].mean() * 100,
            'avg_transaction_amount': data['Transaction_Amount'].mean(),
            'total_transaction_value': data['Transaction_Amount'].sum()
        }
        
        # Time period
        if 'Timestamp' in data.columns:
            data_temp = data.copy()
            data_temp['Timestamp'] = pd.to_datetime(data_temp['Timestamp'])
            summary['time_period'] = {
                'start_date': data_temp['Timestamp'].min(),
                'end_date': data_temp['Timestamp'].max(),
                'duration_days': (data_temp['Timestamp'].max() - data_temp['Timestamp'].min()).days
            }
        
        # Risk analysis
        summary['risk_analysis'] = {
            'high_risk_transactions': len(data[data['Risk_Score'] > 0.7]),
            'avg_risk_score': data['Risk_Score'].mean(),
            'high_risk_fraud_rate': data[data['Risk_Score'] > 0.7]['Fraud_Label'].mean() * 100
        }
        
        # Model performance (if provided)
        if metrics:
            summary['model_performance'] = metrics
        
        return summary
    
    def plot_confusion_matrix_detailed(self, cm, labels=['Legitimate', 'Fraudulent']):
        """
        Create detailed confusion matrix visualization
        
        Args:
            cm: Confusion matrix
            labels: Class labels
        
        Returns:
            Plotly figure
        """
        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create annotations
        annotations = []
        for i in range(len(labels)):
            for j in range(len(labels)):
                annotations.append(
                    dict(
                        x=j, y=i,
                        text=f"{cm[i, j]}<br>({cm_percent[i, j]:.1f}%)",
                        showarrow=False,
                        font=dict(color="white" if cm[i, j] > cm.max()/2 else "black")
                    )
                )
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale='Blues',
            showscale=True
        ))
        
        fig.update_layout(
            title='Detailed Confusion Matrix',
            xaxis_title='Predicted',
            yaxis_title='Actual',
            annotations=annotations
        )
        
        return fig

