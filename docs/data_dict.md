# 数据字典

## 原始数据字段

### 学生信息
| 字段名 | 类型 | 描述 | 示例 |
|-------|------|------|------|
| student_id | string | 学生唯一标识符 | "S2021001" |
| grade | int | 年级 | 2 |
| class | string | 班级 | "2班" |
| major | string | 专业 | "计算机科学" |

### 课程信息
| 字段名 | 类型 | 描述 | 示例 |
|-------|------|------|------|
| course_id | string | 课程唯一标识符 | "C001" |
| course_name | string | 课程名称 | "高等数学" |
| teacher_id | string | 教师ID | "T001" |
| course_type | string | 课程类型 | "必修" |
| credit | float | 学分 | 4.0 |
| course_difficulty | int | 课程难度等级(1-5) | 3 |

### 出勤记录
| 字段名 | 类型 | 描述 | 示例 |
|-------|------|------|------|
| attendance_id | string | 出勤记录ID | "A202109010801" |
| student_id | string | 学生ID | "S2021001" |
| course_id | string | 课程ID | "C001" |
| timestamp | datetime | 上课时间 | "2021-09-01 08:00:00" |
| attendance | int | 出勤状态(0:缺勤,1:出勤) | 1 |
| attendance_type | string | 出勤类型 | "正常出勤" |

## 特征工程后的字段

### 学生历史特征
| 字段名 | 类型 | 描述 | 示例 |
|-------|------|------|------|
| historical_attendance_rate | float | 历史出勤率 | 0.95 |
| attendance_30d_mean | float | 30天平均出勤率 | 0.92 |
| attendance_30d_std | float | 30天出勤率标准差 | 0.08 |
| attendance_30d_max | float | 30天最高出勤率 | 1.0 |
| attendance_30d_min | float | 30天最低出勤率 | 0.8 |

### 时间特征
| 字段名 | 类型 | 描述 | 示例 |
|-------|------|------|------|
| hour | int | 小时 | 8 |
| day | int | 日期 | 15 |
| month | int | 月份 | 9 |
| dayofweek | int | 星期几(0-6) | 2 |
| is_weekend | int | 是否周末 | 0 |
| is_holiday | int | 是否节假日 | 0 |
| is_semester_start | int | 是否学期开始 | 1 |
| is_semester_end | int | 是否学期结束 | 0 |

### 课程特征
| 字段名 | 类型 | 描述 | 示例 |
|-------|------|------|------|
| course_attendance_rate | float | 课程整体出勤率 | 0.88 |
| timeslot_attendance_rate | float | 该时间段出勤率 | 0.85 |
| normalized_difficulty | float | 标准化后的课程难度 | 0.5 |

### 交互特征
| 字段名 | 类型 | 描述 | 示例 |
|-------|------|------|------|
| time_difficulty | float | 时间与难度的交互 | 24.0 |
| day_difficulty | float | 星期与难度的交互 | 6.0 |
| history_time | float | 历史出勤率与时间的交互 | 7.6 |
| history_day | float | 历史出勤率与星期的交互 | 1.9 |

## 目标变量
| 字段名 | 类型 | 描述 | 示例 |
|-------|------|------|------|
| target | int | 是否出勤(0:缺勤,1:出勤) | 1 |

## 数据质量规则
1. 所有ID字段不允许为空
2. timestamp必须是有效的日期时间格式
3. 数值型字段不允许为负
4. 出勤率必须在[0,1]范围内
5. 课程难度必须在[1,5]范围内

## 数据更新频率
- 原始数据：每天更新一次
- 特征工程：每天凌晨2:00执行
- 模型预测：每天早上6:00执行

## 数据存储
- 原始数据：MySQL数据库
- 处理后数据：处理后的CSV文件
- 特征数据：特征存储服务 