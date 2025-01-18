import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional
from pathlib import Path
import json
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

def load_predictions(predictions_path: str) -> pd.DataFrame:
    """加载预测结果"""
    return pd.read_csv(predictions_path)

def save_predictions(predictions: pd.DataFrame, output_path: str,
                    prefix: str = "predictions"):
    """保存预测结果"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(output_path) / f"{prefix}_{timestamp}.csv"
    predictions.to_csv(output_file, index=False)
    logger.info(f"Predictions saved to {output_file}")

def format_predictions(predictions: pd.DataFrame,
                      include_probabilities: bool = True) -> pd.DataFrame:
    """格式化预测结果"""
    formatted = predictions.copy()
    
    # 将预测结果转换为可读的标签
    formatted['prediction'] = formatted['prediction'].map({
        1: "出勤",
        0: "缺勤"
    })
    
    if include_probabilities:
        # 将概率转换为百分比
        formatted['probability'] = formatted['probability'].apply(
            lambda x: f"{x*100:.2f}%"
        )
    else:
        formatted.drop('probability', axis=1, inplace=True)
        
    return formatted

def aggregate_predictions(predictions: pd.DataFrame,
                        groupby_columns: List[str]) -> pd.DataFrame:
    """聚合预测结果"""
    agg_dict = {
        'prediction': 'mean',  # 计算出勤率
        'probability': 'mean'  # 计算平均概率
    }
    
    aggregated = predictions.groupby(groupby_columns).agg(agg_dict)
    aggregated.columns = ['attendance_rate', 'average_probability']
    
    # 将比率转换为百分比格式
    aggregated['attendance_rate'] = aggregated['attendance_rate'].apply(
        lambda x: f"{x*100:.2f}%"
    )
    aggregated['average_probability'] = aggregated['average_probability'].apply(
        lambda x: f"{x*100:.2f}%"
    )
    
    return aggregated.reset_index()

def generate_attendance_report(predictions: pd.DataFrame,
                             metadata: Optional[Dict] = None) -> Dict:
    """生成出勤报告"""
    report = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'summary': {
            'total_records': len(predictions),
            'attendance_rate': f"{predictions['prediction'].mean()*100:.2f}%",
            'average_confidence': f"{predictions['probability'].mean()*100:.2f}%"
        },
        'details': {}
    }
    
    # 添加元数据
    if metadata:
        report['metadata'] = metadata
    
    # 按不同维度统计
    if 'class_id' in predictions.columns:
        class_stats = aggregate_predictions(predictions, ['class_id'])
        report['details']['by_class'] = class_stats.to_dict('records')
        
    if 'student_id' in predictions.columns:
        student_stats = aggregate_predictions(predictions, ['student_id'])
        report['details']['by_student'] = student_stats.to_dict('records')
        
    if 'timestamp' in predictions.columns:
        predictions['date'] = pd.to_datetime(predictions['timestamp']).dt.date
        date_stats = aggregate_predictions(predictions, ['date'])
        report['details']['by_date'] = date_stats.to_dict('records')
    
    return report

def save_report(report: Dict, output_path: str, format: str = 'json'):
    """保存报告"""
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if format == 'json':
        output_file = output_dir / f"attendance_report_{timestamp}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=4)
    elif format == 'csv':
        output_file = output_dir / f"attendance_report_{timestamp}.csv"
        report_df = pd.DataFrame(report['details'])
        report_df.to_csv(output_file, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Report saved to {output_file}")

def analyze_prediction_trends(predictions: pd.DataFrame,
                            time_unit: str = 'day') -> pd.DataFrame:
    """分析预测趋势"""
    predictions = predictions.copy()
    predictions['timestamp'] = pd.to_datetime(predictions['timestamp'])
    
    if time_unit == 'hour':
        predictions['time_group'] = predictions['timestamp'].dt.floor('H')
    elif time_unit == 'day':
        predictions['time_group'] = predictions['timestamp'].dt.date
    elif time_unit == 'week':
        predictions['time_group'] = predictions['timestamp'].dt.to_period('W').astype(str)
    elif time_unit == 'month':
        predictions['time_group'] = predictions['timestamp'].dt.to_period('M').astype(str)
    else:
        raise ValueError(f"Unsupported time unit: {time_unit}")
    
    trends = predictions.groupby('time_group').agg({
        'prediction': 'mean',
        'probability': 'mean',
        'timestamp': 'count'
    }).reset_index()
    
    trends.columns = ['time_group', 'attendance_rate', 'average_confidence', 'count']
    return trends

def calculate_prediction_statistics(predictions: pd.DataFrame) -> Dict:
    """计算预测统计信息"""
    stats = {
        'total_predictions': len(predictions),
        'attendance_count': int(predictions['prediction'].sum()),
        'absence_count': int(len(predictions) - predictions['prediction'].sum()),
        'attendance_rate': f"{predictions['prediction'].mean()*100:.2f}%",
        'average_confidence': f"{predictions['probability'].mean()*100:.2f}%",
        'confidence_std': f"{predictions['probability'].std()*100:.2f}%",
        'high_confidence_predictions': int(
            (predictions['probability'] > 0.9).sum() +
            (predictions['probability'] < 0.1).sum()
        ),
        'low_confidence_predictions': int(
            ((predictions['probability'] >= 0.4) &
             (predictions['probability'] <= 0.6)).sum()
        )
    }
    return stats 