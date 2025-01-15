from .base import BaseModel
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from typing import Dict, Any

class RandomForestModel(BaseModel):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = RandomForestClassifier(
            n_estimators=config.get('n_estimators', 100),
            criterion=config.get('criterion', 'gini'),
            max_depth=config.get('max_depth', None),
            min_samples_split=config.get('min_samples_split', 2),
            min_samples_leaf=config.get('min_samples_leaf', 1),
            random_state=config.get('random_state', 42)
        )
    
    def train(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestModel':
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