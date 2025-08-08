import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import networkx as nx
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Simple Data class to replace torch_geometric.data.Data
class SimpleGraphData:
    def __init__(self, x=None, edge_index=None, edge_attr=None, batch=None):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.batch = batch

class GraphBuilder:
    
    
    def __init__(self, similarity_threshold=0.8, max_neighbors=10):
        self.similarity_threshold = similarity_threshold
        self.max_neighbors = max_neighbors
        self.node_encoder = {}
        self.user_encoder = {}
        self.merchant_encoder = {}
        
    def _encode_entities(self, data):
        
        # Encode users
        unique_users = data['User_ID'].unique()
        self.user_encoder = {user: idx for idx, user in enumerate(unique_users)}
        
        # Encode merchants (using merchant category as proxy)
        unique_merchants = data['Merchant_Category'].unique()
        self.merchant_encoder = {merchant: idx + len(unique_users) 
                               for idx, merchant in enumerate(unique_merchants)}
        
        return len(unique_users), len(unique_merchants)
    
    def _create_user_similarity_edges(self, data, user_features):
        
        edges = []
        
        # Calculate user similarities based on transaction patterns
        user_profiles = data.groupby('User_ID').agg({
            'Transaction_Amount': ['mean', 'std', 'count'],
            'Risk_Score': 'mean',
            'Daily_Transaction_Count': 'mean',
            'Card_Age': 'mean',
            'Transaction_Distance': 'mean'
        }).fillna(0)
        
        # Flatten column names
        user_profiles.columns = ['_'.join(col).strip() for col in user_profiles.columns]
        
        # Calculate similarities
        similarities = cosine_similarity(user_profiles.values)
        
        # Create edges for similar users
        for i in range(len(similarities)):
            for j in range(i + 1, len(similarities)):
                if similarities[i, j] > self.similarity_threshold:
                    user_i = user_profiles.index[i]
                    user_j = user_profiles.index[j]
                    
                    node_i = self.user_encoder[user_i]
                    node_j = self.user_encoder[user_j]
                    
                    edges.append([node_i, node_j])
                    edges.append([node_j, node_i])  
        
        return edges
    
    def _create_user_merchant_edges(self, data):
        
        edges = []
        
        for _, row in data.iterrows():
            user_node = self.user_encoder[row['User_ID']]
            merchant_node = self.merchant_encoder[row['Merchant_Category']]
            
            # Bidirectional edges
            edges.append([user_node, merchant_node])
            edges.append([merchant_node, user_node])
        
        return edges
    
    def _create_temporal_edges(self, data):
        
        edges = []
        
        # Sort by timestamp
        data_sorted = data.sort_values('Timestamp')
        
        # Group by user
        for user_id, user_data in data_sorted.groupby('User_ID'):
            user_node = self.user_encoder[user_id]
            user_transactions = user_data.sort_values('Timestamp')
            
            # Connect consecutive transactions
            for i in range(len(user_transactions) - 1):
                # For temporal edges, we'll connect user nodes with themselves
                # weighted by time difference (implemented via edge attributes)
                edges.append([user_node, user_node])
        
        return edges
    
    def _create_fraud_pattern_edges(self, data):
        
        edges = []
        
        # Connect users with similar fraud risk patterns
        high_risk_users = data[data['Risk_Score'] > 0.7]['User_ID'].unique()
        
        for i, user_i in enumerate(high_risk_users):
            for user_j in high_risk_users[i+1:]:
                if user_i != user_j:
                    node_i = self.user_encoder[user_i]
                    node_j = self.user_encoder[user_j]
                    
                    edges.append([node_i, node_j])
                    edges.append([node_j, node_i])
        
        # Connect users who had transactions in same location at similar times
        location_groups = data.groupby(['Location', 'Timestamp'])['User_ID'].apply(list)
        
        for users_list in location_groups:
            if len(users_list) > 1:
                for i, user_i in enumerate(users_list):
                    for user_j in users_list[i+1:]:
                        node_i = self.user_encoder[user_i]
                        node_j = self.user_encoder[user_j]
                        
                        edges.append([node_i, node_j])
                        edges.append([node_j, node_i])
        
        return edges
    
    def _create_node_features(self, data, processed_features):
        
        num_users = len(self.user_encoder)
        num_merchants = len(self.merchant_encoder)
        total_nodes = num_users + num_merchants
        
        # Feature dimension from processed features
        feature_dim = processed_features.shape[1]
        
        # Initialize node features
        node_features = np.zeros((total_nodes, feature_dim))
        
        # Aggregate features for user nodes
        for user_id, user_data in data.groupby('User_ID'):
            user_node = self.user_encoder[user_id]
            user_indices = user_data.index
            
            # Average features for this user
            user_feature_indices = [i for i, idx in enumerate(data.index) if idx in user_indices]
            if user_feature_indices:
                node_features[user_node] = processed_features[user_feature_indices].mean(axis=0)
        
        # Aggregate features for merchant nodes
        for merchant_cat, merchant_data in data.groupby('Merchant_Category'):
            merchant_node = self.merchant_encoder[merchant_cat]
            merchant_indices = merchant_data.index
            
            # Average features for this merchant
            merchant_feature_indices = [i for i, idx in enumerate(data.index) if idx in merchant_indices]
            if merchant_feature_indices:
                node_features[merchant_node] = processed_features[merchant_feature_indices].mean(axis=0)
        
        return node_features
    
    def _create_edge_attributes(self, edge_list, data):
        
        edge_attrs = []
        
        for edge in edge_list:
            # Simple edge weight (can be enhanced)
            edge_attrs.append([1.0])  # Default weight
        
        return np.array(edge_attrs)
    
    def build_graph(self, data, processed_features):

        # Encode entities
        num_users, num_merchants = self._encode_entities(data)
        
        # Create different types of edges
        user_similarity_edges = self._create_user_similarity_edges(data, processed_features)
        user_merchant_edges = self._create_user_merchant_edges(data)
        temporal_edges = self._create_temporal_edges(data)
        fraud_pattern_edges = self._create_fraud_pattern_edges(data)
        
        # Combine all edges
        all_edges = (user_similarity_edges + user_merchant_edges + 
                    temporal_edges + fraud_pattern_edges)
        
        # Remove duplicate edges
        edge_set = set()
        unique_edges = []
        for edge in all_edges:
            edge_tuple = tuple(edge)
            if edge_tuple not in edge_set:
                edge_set.add(edge_tuple)
                unique_edges.append(edge)
        
        # Convert to tensor
        if unique_edges:
            edge_index = torch.tensor(unique_edges, dtype=torch.long).t().contiguous()
        else:
            # Create empty edge index if no edges
            edge_index = torch.empty((2, 0), dtype=torch.long)
        
        # Create node features
        node_features = self._create_node_features(data, processed_features)
        x = torch.tensor(node_features, dtype=torch.float)
        
        # Create edge attributes
        if unique_edges:
            edge_attr = self._create_edge_attributes(unique_edges, data)
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        else:
            edge_attr = torch.empty((0, 1), dtype=torch.float)
        
        # Create Simple Graph Data object
        graph_data = SimpleGraphData(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr
        )
        
        return graph_data
    
    def create_batch_graphs(self, data_list, processed_features_list):

        graph_list = []
        
        for data, features in zip(data_list, processed_features_list):
            graph = self.build_graph(data, features)
            graph_list.append(graph)
        
        # Simple batching - just return the first graph for now
        # In a full implementation, you'd properly batch the graphs
        if graph_list:
            return graph_list[0]
        else:
            return SimpleGraphData()
    
    def build_transaction_graph(self, data, processed_features):

        num_transactions = len(data)
        
        # Use processed features directly as node features
        x = torch.tensor(processed_features, dtype=torch.float)
        
        # Create edges based on various criteria
        edges = []
        
        # Connect transactions from same user
        user_transactions = data.groupby('User_ID').groups
        for user_id, txn_indices in user_transactions.items():
            txn_list = list(txn_indices)
            for i in range(len(txn_list)):
                for j in range(i + 1, min(i + 5, len(txn_list))):  # Connect to next 4 transactions
                    edges.append([txn_list[i], txn_list[j]])
                    edges.append([txn_list[j], txn_list[i]])
        
        # Connect transactions with similar amounts
        amounts = data['Transaction_Amount'].values
        for i in range(num_transactions):
            similar_amount_mask = np.abs(amounts - amounts[i]) < (0.1 * amounts[i])
            similar_indices = np.where(similar_amount_mask)[0]
            
            for j in similar_indices[:5]:  # Limit connections
                if i != j:
                    edges.append([i, j])
        
        # Connect transactions in same location and time window
        data_with_indices = data.reset_index()
        for location in data['Location'].unique():
            location_data = data_with_indices[data_with_indices['Location'] == location]
            
            if len(location_data) > 1:
                # Sort by timestamp
                location_data = location_data.sort_values('Timestamp')
                
                for i in range(len(location_data) - 1):
                    for j in range(i + 1, min(i + 3, len(location_data))):
                        idx_i = location_data.iloc[i]['index']
                        idx_j = location_data.iloc[j]['index']
                        edges.append([idx_i, idx_j])
                        edges.append([idx_j, idx_i])
        
        # Remove duplicates and convert to tensor
        edge_set = set()
        unique_edges = []
        for edge in edges:
            edge_tuple = tuple(edge)
            if edge_tuple not in edge_set:
                edge_set.add(edge_tuple)
                unique_edges.append(edge)
        
        if unique_edges:
            edge_index = torch.tensor(unique_edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        
        # Create graph
        graph_data = SimpleGraphData(
            x=x,
            edge_index=edge_index
        )
        
        return graph_data
    
    def analyze_graph_properties(self, graph_data):

        num_nodes = graph_data.x.size(0)
        num_edges = graph_data.edge_index.size(1)
        
        # Convert to NetworkX for analysis
        edge_list = graph_data.edge_index.t().numpy()
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        G.add_edges_from(edge_list)
        
        properties = {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'average_degree': 2 * num_edges / num_nodes if num_nodes > 0 else 0,
            'num_connected_components': nx.number_connected_components(G),
            'largest_component_size': len(max(nx.connected_components(G), key=lambda x: len(x))) if num_nodes > 0 else 0,
            'clustering_coefficient': nx.average_clustering(G) if num_nodes > 0 else 0,
            'density': nx.density(G)
        }
        
        return properties
