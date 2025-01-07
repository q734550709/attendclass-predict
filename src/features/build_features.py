import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta
import holidays

logger = logging.getLogger(__name__)

class FeatureBuilder:
    def __init__(self, config: Dict):
        """初始化特征构建器"""
        self.config = config
        self.cn_holidays = holidays.CN()  # 中国节假日日历
        
    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """构建所有特征"""
        # 构建学生历史特征
        df = self.build_student_history_features(df)
        
        # 构建时间特征
        df = self.build_temporal_features(df)
        
        # 构建课程特征
        df = self.build_course_features(df)
        
        # 构建交互特征
        if self.config['features'].get('interaction', {}).get('enable', False):
            df = self.build_interaction_features(df)
            
        return df
        
    def build_student_history_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """构建学生历史特征"""
        window = self.config['features']['student_history']['window_size']
        aggregations = self.config['features']['student_history']['aggregations']
        
        # 按学生ID和时间排序
        df = df.sort_values(['student_id', 'timestamp'])
        
        # 计算历史出勤率
        df['historical_attendance_rate'] = df.groupby('student_id')['attendance'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        
        # 计算其他统计特征
        for agg in aggregations:
            if agg == 'mean':
                df[f'attendance_{window}d_mean'] = df.groupby('student_id')['attendance'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
            elif agg == 'std':
                df[f'attendance_{window}d_std'] = df.groupby('student_id')['attendance'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).std()
                )
            elif agg == 'max':
                df[f'attendance_{window}d_max'] = df.groupby('student_id')['attendance'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).max()
                )
            elif agg == 'min':
                df[f'attendance_{window}d_min'] = df.groupby('student_id')['attendance'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).min()
                )
                
        return df
        
    def build_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """构建时间特征"""
        # 确保timestamp列为datetime类型
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 提取基本时间特征
        df['hour'] = df['timestamp'].dt.hour
        df['day'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['dayofweek'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
        
        # 添加节假日特征
        if self.config['features']['temporal'].get('add_holidays', False):
            df['is_holiday'] = df['timestamp'].dt.date.map(
                lambda x: 1 if x in self.cn_holidays else 0
            )
            
        # 添加学期特征
        df['is_semester_start'] = df.apply(self._is_semester_start, axis=1)
        df['is_semester_end'] = df.apply(self._is_semester_end, axis=1)
        
        return df
        
    def build_course_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """构建课程相关特征"""
        # 计算每门课程的历史出勤率
        df['course_attendance_rate'] = df.groupby('course_id')['attendance'].transform('mean')
        
        # 计算每个时间段的出勤率
        df['timeslot_attendance_rate'] = df.groupby('hour')['attendance'].transform('mean')
        
        # 计算课程难度指标（如果有相关数据）
        if 'course_difficulty' in df.columns:
            df['normalized_difficulty'] = df.groupby('course_id')['course_difficulty'].transform(
                lambda x: (x - x.mean()) / x.std()
            )
            
        return df
        
    def build_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """构建特征交互"""
        max_degree = self.config['features']['interaction'].get('max_degree', 2)
        
        # 一阶交互
        df['time_difficulty'] = df['hour'] * df.get('course_difficulty', 0)
        df['day_difficulty'] = df['dayofweek'] * df.get('course_difficulty', 0)
        
        if max_degree >= 2:
            # 二阶交互
            df['history_time'] = df['historical_attendance_rate'] * df['hour']
            df['history_day'] = df['historical_attendance_rate'] * df['dayofweek']
            
        return df
        
    def _is_semester_start(self, row) -> int:
        """判断是否为学期开始时间"""
        # 这里需要根据实际的学期时间进行调整
        month = row['timestamp'].month
        day = row['timestamp'].day
        
        # 假设春季学期2月底开始，秋季学期9月初开始
        if (month == 2 and day >= 20) or (month == 9 and day <= 10):
            return 1
        return 0
        
    def _is_semester_end(self, row) -> int:
        """判断是否为学期结束时间"""
        # 这里需要根据实际的学期时间进行调整
        month = row['timestamp'].month
        day = row['timestamp'].day
        
        # 假设春季学期6月底结束，秋季学期1月初结束
        if (month == 6 and day >= 20) or (month == 1 and day <= 10):
            return 1
        return 0 