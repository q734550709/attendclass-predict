import os
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from pathlib import Path # Path的作用是将字符串转换为路径
import yaml # 用于加载配置文件
import logging # 用于打印日志
from typing import Dict, Tuple, List, Any # 用于类型提示

import warnings
# 抑制所有警告信息
warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict:
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_raw_data(data_path: str, columns_to_read: list, chunk_size: int, sample_frac: float, random_state: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """加载原始数据"""

    # 设置抽样和未抽样的了数据列表
    sampled_chunks = []
    unsampled_chunks = []

    # 去掉列名中的表名前缀
    sanitized_columns = [col[2:] if col.startswith('t.') else col for col in columns_to_read]

    try:
        for chunk in pd.read_csv(data_path, chunksize=chunk_size, usecols=columns_to_read):
            # 去掉列名中的表名前缀
            chunk.columns = sanitized_columns
            # 抽样
            sampled_chunk = chunk.sample(frac=sample_frac, random_state=random_state)
            # 未被抽样的数据
            unsampled_chunk = chunk.drop(sampled_chunk.index)
            sampled_chunks.append(sampled_chunk)
            unsampled_chunks.append(unsampled_chunk)
    except ValueError:
        for chunk in pd.read_csv(data_path, chunksize=chunk_size, usecols=sanitized_columns):
            # 抽样
            sampled_chunk = chunk.sample(frac=sample_frac, random_state=random_state)
            # 未被抽样的数据
            unsampled_chunk = chunk.drop(sampled_chunk.index)
            sampled_chunks.append(sampled_chunk)
            unsampled_chunks.append(unsampled_chunk)

    # 合并所有抽样的数据和未抽样的数据
    sampled_df = pd.concat(sampled_chunks, ignore_index=True)
    unsampled_df = pd.concat(unsampled_chunks, ignore_index=True)

    logger.info(f"Loaded raw data from {data_path} with {len(sampled_df)} sampled rows and {len(unsampled_df)} unsampled rows")
    return sampled_df, unsampled_df

def calculate_label(df: pd.DataFrame, target_column: str, label_columns: list) -> pd.DataFrame:
    # 在进行数据平衡之前计算label
    df[target_column] = df[label_columns].fillna(0).prod(axis=1)
    return df

def balance_and_merge_data(sampled_df: pd.DataFrame, unsampled_df: pd.DataFrame, target_column: str = 'label', test_size: float = 0.3, random_state: int = 42) -> pd.DataFrame:
    """平衡并合并数据"""
    # 进行训练集和测试集划分
    X = sampled_df.drop(target_column, axis=1)
    y = sampled_df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # 检查训练集中的类别分布
    train_class_counts = y_train.value_counts()
    min_class = train_class_counts.idxmin()
    max_class = train_class_counts.idxmax()
    diff = train_class_counts[max_class] - train_class_counts[min_class]

    if diff > 0:
        # 从未抽样的数据中获取少数类样本
        unsampled_minority = unsampled_df[unsampled_df[target_column] == min_class]
        # 如果未抽样的数据中少数类样本不足，取全部
        additional_samples = unsampled_minority.sample(n=min(diff, len(unsampled_minority)), random_state=random_state)

        # 将补充的少数类样本添加到训练集中
        X_additional = additional_samples.drop(target_column, axis=1)
        y_additional = additional_samples[target_column]
        X_train = pd.concat([X_train, X_additional], ignore_index=True)
        y_train = pd.concat([y_train, y_additional], ignore_index=True)

    # 合并训练集和测试集
    X_train['is_train_dataset'] = 1
    X_test['is_train_dataset'] = 0
    raw_data = pd.concat([X_train, X_test], ignore_index=True)
    raw_data[target_column] = pd.concat([y_train, y_test], ignore_index=True)
    logger.info(f"Merged balanced data with {len(raw_data)} rows")

    return raw_data

def create_features(df: pd.DataFrame, new_feature_config: Dict) -> pd.DataFrame:
    """新增字段的函数"""
    for new_feature in new_feature_config:
        calculation_type = new_feature['calculation_type']
        cols = new_feature['cols']
        new_col = new_feature['new_col']
        
        if calculation_type == 'division':
            df[new_col] = df[cols[0]] / df[cols[1]]
        elif calculation_type == 'is_not_null':
            df[new_col] = df[cols[0]].notna().astype('float32')
        elif calculation_type == 'sum':
            df[new_col] = df[cols].sum(axis=1)
        elif calculation_type == 'datetime_to_timestamp':
            df[new_col] = pd.to_datetime(df[cols[0]], errors='coerce').astype('int64') / 10**9
        else:
            raise ValueError(f"Unknown calculation type: {calculation_type}")
    
    logger.info(f"Created {len(new_feature_config)} new features")
    return df

def handle_null_values(df: pd.DataFrame, null_indicators: Dict) -> pd.DataFrame:
    """处理null值或者0对应one-hot编码"""
    for column, value in null_indicators.items():
        new_column_name = f"{column}_{value}"
        
        if value == 'null':
            # 如果值为null，则创建一个新列，标记原列中是否为null
            df[new_column_name] = df[column].isnull().astype(int)
        else:
            # 如果值不为null，则创建一个新列，标记原列中是否等于指定值
            df[new_column_name] = (df[column] == value).astype(int)
    
    logger.info(f"Created {len(null_indicators)} null indicator columns")
    return df

def handle_missing_values(df: pd.DataFrame, fill_values: Dict) -> pd.DataFrame:
    """处理缺失值"""
    for column, value in fill_values.items():
        if value == '中位数':
            # 使用中位数填充缺失值
            df[column] = df[column].fillna(df[column].median())
        else:
            # 使用指定值填充缺失值
            df[column] = df[column].fillna(value)
    logger.info(f"Filled missing values for {len(fill_values)} columns")
    return df

def process_unusual_features(df: pd.DataFrame, feature_dict: Dict[str, str]) -> pd.DataFrame:
    """处理异常特征"""
    for feature, condition in feature_dict.items():
        if condition == 'less_than_zero':
            # 如果特征值小于0，则将其设置为-1
            df[feature] = np.where(df[feature] < 0, -1, df[feature])
        elif condition == 'greater_than_one':
            # 如果特征值大于1，则将其设置为1
            df[feature] = np.where(df[feature] > 1, 1, df[feature])
    logger.info(f"Processed {len(feature_dict)} unusual features")
    return df

def onehot_greater_than_one(df: pd.DataFrame, feature_list: list) -> pd.DataFrame:
    """将特征值大于1的one-hot编码为0"""
    df[feature_list] = df[feature_list].applymap(lambda x: 0 if x > 1 else x)
    logger.info(f"Processed {len(feature_list)} features with values greater than 1")
    return df

def one_hot_encode_with_custom_categories(df: pd.DataFrame, mapping_dict: Dict[str, Dict[int, str]], categories_dict: Dict[str, list]) -> pd.DataFrame:
    """使用自定义类别进行 one-hot 编码"""
    # 遍历映射字典，将数字编码转换为描述性字符串编码
    for column, mapping in mapping_dict.items():
        df[column] = df[column].map(mapping)
    
    # 对于 categories_dict 中定义的类别，设置 pd.Categorical 并进行 one-hot 编码
    for column, categories in categories_dict.items():
        # 创建 Categorical 列，确保包含所有类别
        df[column] = pd.Categorical(df[column], categories=categories)
        
        # 对转换后的列进行 one-hot 编码
        df = pd.get_dummies(df, columns=[column], prefix=column)
    logger.info(f"Encoded {len(mapping_dict) + len(categories_dict)} columns with custom categories")
    return df

def split_data(df: pd.DataFrame, config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """划分数据集"""
    logger.info("Splitting data into train/test sets")
    # 获取划分数据集的配置
    is_train_label = config['is_train_label']
    y_label = config['y_label']

    # 划分数据集
    train_df, test_df = df[df[is_train_label]==1], df[df[is_train_label]==0]
    X_train, X_test, y_train, y_test = train_df.drop(y_label, axis=1), test_df.drop(y_label, axis=1), train_df[y_label], test_df[y_label]
    return X_train, X_test, y_train, y_test

def save_data(output_df: pd.DataFrame, output_path: str, output_name: str) -> None:
    """保存处理后的数据"""
    logger.info(f"Saving data to {output_path} with name {output_name}")
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    output_df.to_csv(f"{output_path}/{output_name}.csv", index=False)


def main():
    """主函数"""
    # 加载数据配置
    config = load_config("../../configs/base/data.yaml")
    
    # 获取raw_data文件列表
    folder_path = config['paths']['raw']
    # 遍历文件夹下所有文件，且只获取csv文件，开头不是'._'的文件
    csv_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv') and not file.startswith('._')]

    """加载原始数据需要的参数"""
    # 获取数据处理的配置
    data_processing_config = config['data_processing_config']
    chunk_size = data_processing_config['chunk_size']
    sample_frac = data_processing_config['sample_frac']
    random_state = data_processing_config['random_state']
    test_size = data_processing_config['test_size']
    target_column = data_processing_config['target_column']

    # 获取需要读取的列
    columns_to_read = config['columns_to_read']
    # 获取label计算所需的列
    label_columns = config['label_columns']
    # 获取raw_data的新列名
    new_columns = config['new_columns']

    """数据清洗需要的参数"""
    # 获取features的配置
    new_feature_config = config['new_features']
    # 获取null值和0的one-hot编码
    null_indicators = config['null_indicators']
    # 获取缺失值填充的值
    fill_values = config['fill_values']
    # 获取异常值处理的配置
    unusual_feature_dict = config['unusual_feature_dict']
    # 获取特征值大于1的特征列表
    onehot_greater_than_one_features = config['onehot_greater_than_one_features']
    # 获取自定义类别的映射字典和类别字典
    mapping_dict = config['mapping_dict']
    categories_dict = config['categories_dict']
    # 获取需要删除的列
    columns_to_drop = config['columns_to_drop']


    # 加载原始数据
    raw_data_list = []
    # 遍历每个文件，调用函数并将结果添加到相应列表
    for csv_file in csv_files:
        # 加载原始数据，返回抽样和未抽样的数据
        sampled_df, unsampled_df = load_raw_data(csv_file, columns_to_read, chunk_size, sample_frac, random_state)    
    
        # 计算label
        sampled_df = calculate_label(sampled_df, target_column, label_columns)
        unsampled_df = calculate_label(unsampled_df, target_column, label_columns)

        # 平衡并合并数据
        raw_data = balance_and_merge_data(sampled_df, unsampled_df, target_column, test_size, random_state)

        # 将处理后的数据添加到列表
        raw_data_list.append(raw_data)

    # 合并数据
    raw_data = pd.concat(raw_data_list, ignore_index=True)
    # 重命名列
    raw_data.columns = new_columns

    # 新增字段，并创建新的数据集dataset_df
    dataset_df = create_features(raw_data, new_feature_config)

    # 增加null和0的one-hot编码
    dataset_df = handle_null_values(dataset_df, null_indicators)

    # 处理缺失值
    dataset_df = handle_missing_values(dataset_df, fill_values)

    # 异常值处理
    # 特殊情况处理
    # 总退费次数:>0 的改为1
    dataset_df['总退费次数'] = np.where(dataset_df['总退费次数'] > 0, 1, dataset_df['总退费次数'])
    # 年龄：<4或者>13的年龄记为0
    dataset_df['年龄'] = np.where((dataset_df['年龄'] > 13) | (dataset_df['年龄'] < 4), 0, dataset_df['年龄'])
    # 班级最大学员数量_one：非6和8使用0替换
    dataset_df['班级最大学员数量_one'] = np.where((dataset_df['班级最大学员数量_one'] != 6) & (dataset_df['班级最大学员数量_one'] != 8), 0, dataset_df['班级最大学员数量_one'])

    # 处理异常特征
    dataset_df = process_unusual_features(dataset_df, unusual_feature_dict)

    # 特征值大于1的one-hot编码为0
    dataset_df = onehot_greater_than_one(dataset_df, onehot_greater_than_one_features)

    # 使用自定义类别进行 one-hot 编码
    dataset_df = one_hot_encode_with_custom_categories(dataset_df, mapping_dict, categories_dict)

    # 剔除多余字段
    cleaned_df = dataset_df.drop(columns=columns_to_drop)
    # 更新字段类型到float32
    cleaned_df = cleaned_df.astype('float32')

    """分割数据集成训练集和测试集"""
    # 获取特征需要删除的列
    feature_drop_columns = config['split_data_config']['feature_drop_columns']
    # 读取处理后的数据,并删除不需要的列
    feature_data = cleaned_df.drop(columns = feature_drop_columns)
    # 获取划分数据集的配置
    split_data_config = config['split_data_config']
    # 划分数据集
    X_train, X_test, y_train, y_test = split_data(feature_data, split_data_config)

    """将数据输出"""
    # 获取输出文件路径和文件名
    interim_file_path = config['paths']['interim']
    interim_file_name = config['output_config']['interim_file_name']
    # 将预处理的数据保存到interim文件夹下
    save_data(cleaned_df, interim_file_path, interim_file_name)

    # 获取保存处理后的数据的路径
    feature_file_path = config['paths']['processed']
    process_file_name = config['output_config']['process_file_name']
    
    # 保存处理后的数据
    for data, name in zip([X_train, X_test, y_train, y_test], process_file_name):
        save_data(data, feature_file_path, name)

    logger.info("Data processing completed")

if __name__ == "__main__":
    main() 