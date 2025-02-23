import torch

class FloodPredictor:
    def __init__(self, use_gpu=True):
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        
        # Updated realistic bounds (you may adjust these based on your data)
        self.realistic_bounds = {
            'Max_Temp': (25, 44),      
            'Min_Temp': (6, 28),       
            'Rainfall': (0, 2072),
            'Relative_Humidity': (30, 100),
            'Wind_Speed': (0, 20),     
            'Cloud_Coverage': (0, 1),
            # 'Month': (1, 12)
        }
        
        self.thresholds = {k: {
            'low': torch.tensor(v[0] + (v[1] - v[0]) * 0.3, device=self.device),
            'medium': torch.tensor(v[0] + (v[1] - v[0]) * 0.5, device=self.device),
            'high': torch.tensor(v[0] + (v[1] - v[0]) * 0.7, device=self.device)
        } for k, v in self.realistic_bounds.items()}
        
        self.delta = {
            'Max_Temp': 1.0,
            'Min_Temp': 1.0,
            'Rainfall': 10.0,
            'Relative_Humidity': 5.0,
            'Wind_Speed': 0.5,
            'Cloud_Coverage': 0.1,
        }
        
        self.num_bins = 20
        self.num_actions = 3
        self.Q_tables = {
            feature: {
                level: torch.zeros((self.num_bins, self.num_actions), device=self.device)
                for level in ['low', 'medium', 'high']
            } for feature in self.realistic_bounds.keys()
        }
        
        self.history = {
            'accuracy': [],
            'thresholds': []
        }
    def enforce_threshold_order(self, feature):
        """Ensure thresholds maintain low < medium < high order"""
        min_val = torch.tensor(self.realistic_bounds[feature][0], 
                              dtype=torch.float32, 
                              device=self.device)
        max_val = torch.tensor(self.realistic_bounds[feature][1], 
                              dtype=torch.float32, 
                              device=self.device)

        # Clamp low threshold between min_val and medium
        self.thresholds[feature]['low'] = torch.clamp(
            self.thresholds[feature]['low'], 
            min_val,
            self.thresholds[feature]['medium']
        )

        # Clamp medium threshold between low and high
        self.thresholds[feature]['medium'] = torch.clamp(
            self.thresholds[feature]['medium'],
            self.thresholds[feature]['low'],
            self.thresholds[feature]['high']
        )
    
        # Clamp high threshold between medium and max_val
        self.thresholds[feature]['high'] = torch.clamp(
            self.thresholds[feature]['high'],
            self.thresholds[feature]['medium'],
            max_val
        )
    def smooth_score(self, feature_data, thresholds_dict, weight, k=0.05):
        """
        Computes a smooth score using a sigmoid function around each threshold.
        """
        s_low = torch.sigmoid(k * (feature_data - thresholds_dict['low']))
        s_med = torch.sigmoid(k * (feature_data - thresholds_dict['medium']))
        s_high = torch.sigmoid(k * (feature_data - thresholds_dict['high']))
        return weight * (s_low + s_med + s_high)
    
    def compute_feature_interactions(self, batch_data):
        interactions = {}
        interactions['rainfall_humidity'] = (batch_data['Rainfall'] * batch_data['Relative_Humidity'] / 100)
        interactions['temp_range'] = (batch_data['Max_Temp'] - batch_data['Min_Temp'])
        interactions['rainfall_wind'] = (batch_data['Rainfall'] * batch_data['Wind_Speed'])
        interactions['cloud_humidity'] = (batch_data['Cloud_Coverage'] * batch_data['Relative_Humidity'])
        return interactions
    def normalize_feature(self, feature_data, feature_name):
        min_val, max_val = self.realistic_bounds[feature_name]
        return (feature_data - min_val) / (max_val - min_val)

    def rule_based_flood_probability(self, batch_data):
        batch_size = batch_data['Rainfall'].shape[0]
        scores = torch.zeros(batch_size, device=self.device)
        normalized_data = {
        feature: self.normalize_feature(data, feature)
        for feature, data in batch_data.items()
        if feature in self.realistic_bounds
    }
        interactions = self.compute_feature_interactions(normalized_data)
        
        # Define weights for each feature
        feature_weights = {
            'Rainfall': 4.0,
            'Relative_Humidity': 2.0,
            'Wind_Speed': 1.0,
            'Cloud_Coverage': 0.8,
            'Max_Temp': 0.5,
            'Min_Temp': 2.0,
        }
        
        for feature, weight in feature_weights.items():
            scores += self.smooth_score(batch_data[feature], self.thresholds[feature], weight, k=0.05)
        
        scores += torch.where(
            interactions['rainfall_humidity'] > self.compute_adaptive_threshold(interactions['rainfall_humidity']),
            2 * torch.ones(batch_size, device=self.device),
            torch.zeros(batch_size, device=self.device)
        )
        scores += torch.where(
            interactions['rainfall_wind'] > self.compute_adaptive_threshold(interactions['rainfall_wind']),
            1.5 * torch.ones(batch_size, device=self.device),
            torch.zeros(batch_size, device=self.device)
        )
        
        max_possible_score = 30
        return torch.clamp(scores / max_possible_score, 0, 1)
    
    def prepare_batch_data(self, df, batch_indices):
        return {
            col: torch.tensor(df[col].values[batch_indices], 
                            dtype=torch.float32, 
                            device=self.device)
            for col in df.columns if col != 'Flood?'
        }
    
    def discretize_threshold(self, value, min_val, max_val):
        value = torch.tensor(value, device=self.device) if not torch.is_tensor(value) else value
        value = torch.clamp(value, min_val, max_val)
        bin_width = (max_val - min_val) / self.num_bins
        state = ((value - min_val) / bin_width).long()
        return torch.clamp(state, 0, self.num_bins - 1)
    
    def compute_adaptive_threshold(self, values):
        if values.numel() > 1:
            return values.mean() + values.std()
        else:
            return values.mean()
    
    def compute_accuracy(self, batch_data):
        probs = self.rule_based_flood_probability(batch_data)
        predictions = (probs > 0.5).float()
        return (predictions == batch_data['Flood?']).float().mean()
    def validate_prediction(self, features, prediction, actual=None):
    # Convert tensor values to float for comparison
        rainfall = features['Rainfall'].item()
        humidity = features['Relative_Humidity'].item()
        min_temp = features['Min_Temp'].item()

        # Check if prediction makes sense given the features
        high_risk = (
            rainfall > 500 and 
            humidity > 70 and 
            min_temp > 20
        )
        low_risk = (
            rainfall < 100 and 
            humidity < 50
        )

        if high_risk and prediction < 0.5:
            print("Warning: Predicted no flood despite high-risk conditions")
        elif low_risk and prediction > 0.5:
            print("Warning: Predicted flood despite low-risk conditions")

        if actual is not None:
            print(f"Predicted: {prediction:.3f}, Actual: {actual}")
            # Convert tensor values to float before formatting
            formatted_features = {
                k: f"{v.item():.2f}" if torch.is_tensor(v) else f"{v:.2f}" 
                for k, v in features.items()
            }
            print("Features:", formatted_features)
    def train(self, df, num_episodes=100000, batch_size=128, alpha=0.1, gamma=0.9, initial_epsilon=0.1):
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
        
        best_accuracy = 0
        best_thresholds = None
        patience = 1000
        no_improve_count = 0
        min_epsilon = 0.01
        decay_rate = 0.9995
        
        for episode in range(num_episodes):
            current_epsilon = max(initial_epsilon * (decay_rate ** episode), min_epsilon)
            
            batch_indices = np.random.choice(len(train_df), batch_size)
            batch_data = self.prepare_batch_data(train_df, batch_indices)
            batch_data['Flood?'] = torch.tensor(
                train_df['Flood?'].values[batch_indices],
                dtype=torch.float32,
                device=self.device
            )
            
            episode_baseline = self.compute_accuracy(batch_data)
            
            for feature in self.thresholds:
                min_val, max_val = self.realistic_bounds[feature]
                for level in ['low', 'medium', 'high']:
                    current_threshold = self.thresholds[feature][level]
                    state = self.discretize_threshold(current_threshold, min_val, max_val)
                    
                    if torch.rand(1, device=self.device) < current_epsilon:
                        action_idx = torch.randint(0, self.num_actions, (1,), device=self.device)
                    else:
                        action_idx = self.Q_tables[feature][level][state].argmax()
                    
                    actions_tensor = torch.tensor([-1, 0, 1], device=self.device)
                    new_threshold = current_threshold + actions_tensor[action_idx] * self.delta[feature]
                    self.thresholds[feature][level] = torch.clamp(new_threshold, min_val, max_val)
                    
                    new_accuracy = self.compute_accuracy(batch_data)
                    reward = new_accuracy - episode_baseline
                    if feature == 'Rainfall' and self.thresholds[feature][level] < 50:
                        reward -= 5
                    new_state = self.discretize_threshold(self.thresholds[feature][level], min_val, max_val)
                    
                    self.Q_tables[feature][level][state, action_idx] += alpha * (
                        reward + gamma * self.Q_tables[feature][level][new_state].max() -
                        self.Q_tables[feature][level][state, action_idx]
                    )
            for feature in self.thresholds:
                self.enforce_threshold_order(feature)
                
            if (episode + 1) % 1000 == 0:
                val_data = self.prepare_batch_data(val_df, np.arange(len(val_df)))
                val_data['Flood?'] = torch.tensor(
                    val_df['Flood?'].values,
                    dtype=torch.float32,
                    device=self.device
                )
                val_accuracy = self.compute_accuracy(val_data)
                
                self.history['accuracy'].append(val_accuracy.item())
                self.history['thresholds'].append(
                    {k: {kk: vv.item() for kk, vv in v.items()} 
                     for k, v in self.thresholds.items()}
                )
                
                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    best_thresholds = {k: {kk: vv.clone() for kk, vv in v.items()} 
                                     for k, v in self.thresholds.items()}
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                
                if no_improve_count >= patience:
                    print(f"Early stopping at episode {episode+1}")
                    self.thresholds = best_thresholds
                    break
                
                print(f"Episode {episode+1}, Train Accuracy: {episode_baseline:.4f}, "
                      f"Val Accuracy: {val_accuracy:.4f}")
                print(self.history['thresholds'][-1])