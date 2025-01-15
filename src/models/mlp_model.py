from sklearn.neural_network import MLPClassifier
from .base import BaseModel
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from typing import Dict, Any

class MLPModel(BaseModel):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = MLPClassifier(
            hidden_layer_sizes=config.get('hidden_layer_sizes', (100,)),
            activation=config.get('activation', 'relu'),
            solver=config.get('solver', 'adam'),
            alpha=config.get('alpha', 0.0001),
            batch_size=config.get('batch_size', 'auto'),
            learning_rate=config.get('learning_rate', 'constant'),
            learning_rate_init=config.get('learning_rate_init', 0.001),
            max_iter=config.get('max_iter', 200),
            random_state=config.get('random_state', None)
        )
    
    def train(self, X: np.ndarray, y: np.ndarray) -> 'MLPModel':
        self.model.fit(X, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        y_pred = self.predict(X)
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        roc_auc = roc_auc_score(y, y_pred)
        cm = confusion_matrix(y, y_pred)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm
        }
