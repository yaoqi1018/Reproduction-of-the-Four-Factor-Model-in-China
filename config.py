"""
中国四因子模型 CH4 配置文件
"""

DATA_DIR = "原始数据"
OUTPUT_DIR = "output"
OFFICIAL_DIR = "四因子官方数据"

# 数据文件路径
RETURN_FILE = f"{DATA_DIR}/月个股回报率.csv"
RISKFREE_FILE = f"{DATA_DIR}/无风险利率.csv"
COMPANY_INFO_FILE = f"{DATA_DIR}/公司基本情况表.csv"
BALANCE_SHEET_FILE = f"{DATA_DIR}/资产负债表.csv"
INCOME_STATEMENT_FILE = f"{DATA_DIR}/利润表.csv"
SUSPENSION_FILE = f"{DATA_DIR}/停复牌信息.csv"
TURNOVER_FILE = f"{DATA_DIR}/换手率.csv"

# 官方因子数据
OFFICIAL_FACTOR_FILE = f"{OFFICIAL_DIR}/CH4_factors_monthly_202602.csv"

# 因子构建参数
MIN_LISTED_MONTHS = 6          # 剔除上市不满6个月的新股
MIN_SUSP_DAYS = 15             # 月内停牌天数超过此阈值则剔除
ST_FLAG = True                 # 是否剔除ST股
BM_LAG_MONTHS = 6              # 账面市值比滞后月份数

# 单位换算：Msmvttl通常为千元，资产负债表为元，统一换算到元
# 如果Msmvttl单位已经是元，设为1
MSMVTTL_SCALE = 1000           # Msmvttl 单位放大倍数（千元→元）

# BM分组断点
BM_HIGH_PCT = 0.30             # 高BM组（前30%）
BM_MID_PCT = 0.40              # 中BM组
BM_LOW_PCT = 0.30              # 低BM组（后30%）

# 规模分组断点
SIZE_SPLIT_PCT = 0.50          # 规模中位数分组

# 情绪因子(PM)参数
PM_SHELL_PCT = 0.30             # 剔除市值最小30%壳股
PM_LOW_PCT = 0.30               # 低换手率组（前30%）
PM_MID_PCT = 0.40               # 中换手率组
PM_HIGH_PCT = 0.30              # 高换手率组（后30%）
PM_MIN_TURN_MONTHS = 6          # 异常换手率分母最少需要的月份数

# 市值权重选择
USE_CIRCULATING_MV = True       # 组合内加权：True=流通市值加权

# 时间范围
START_DATE = "2010-01-01"
END_DATE = "2026-04-30"

# 输出文件
FACTOR_OUTPUT = f"{OUTPUT_DIR}/four_factors_monthly.csv"
FACTOR_STATS = f"{OUTPUT_DIR}/factor_statistics.csv"
ALPHA_OUTPUT = f"{OUTPUT_DIR}/csi300_alpha.csv"
