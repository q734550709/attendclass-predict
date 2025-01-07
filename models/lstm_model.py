import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from .base import BaseModel

class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM前向传播
        out, _ = self.lstm(x, (h0, c0))
        
        # 解码最后一个时间步
        out = self.fc(out[:, -1, :])
        return out

class LSTMModel(BaseModel):
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = LSTMNet(
            input_size=config.get('input_size', 10),
            hidden_size=config.get('hidden_size', 64),
            num_layers=config.get('num_layers', 2),
            num_classes=config.get('num_classes', 2)
        ).to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=config.get('learning_rate', 0.001)
        )
        
    def _prepare_data(self, X, y=None):
        """将数据转换为PyTorch张量"""
        X = torch.FloatTensor(X).to(self.device)
        if y is not None:
            y = torch.LongTensor(y).to(self.device)
            return TensorDataset(X, y)
        return X
    
    def train(self, X, y):
        """训练LSTM模型
        
        Args:
            X: 特征矩阵 [samples, time_steps, features]
            y: 目标变量
        """
        dataset = self._prepare_data(X, y)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.config.get('batch_size', 32),
            shuffle=True
        )
        
        num_epochs = self.config.get('num_epochs', 10)
        self.model.train()
        
        for epoch in range(num_epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                
            if (epoch + 1) % 5 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}')
        
        return self
    
    def predict(self, X):
        """预测类别
        
        Args:
            X: 特征矩阵 [samples, time_steps, features]
            
        Returns:
            预测的类别标签
        """
        self.model.eval()
        X = self._prepare_data(X)
        with torch.no_grad():
            outputs = self.model(X)
            _, predicted = torch.max(outputs.data, 1)
        return predicted.cpu().numpy()
    
    def predict_proba(self, X):
        """预测概率
        
        Args:
            X: 特征矩阵
            
        Returns:
            预测的概率值
        """
        self.model.eval()
        X = self._prepare_data(X)
        with torch.no_grad():
            outputs = self.model(X)
            probabilities = torch.softmax(outputs, dim=1)
        return probabilities.cpu().numpy()
    
    def evaluate(self, X, y):
        """评估模型性能
        
        Args:
            X: 特征矩阵
            y: 真实标签
            
        Returns:
            dict: 包含准确率等评估指标的字典
        """
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)
        return {
            'accuracy': accuracy
        } 