"""
数据预处理模块：筛选、清洗、标记
"""

import pandas as pd
import numpy as np
from config import *


def preprocess_returns(df_ret, df_company, df_susp):
    """
    个股回报率数据预处理：
    1. 标记上市不足6个月的新股
    2. 标记月内停牌过久的股票
    3. 标记ST股（通过股票名称）
    返回清洗后的数据及各项筛除统计
    """
    print("=" * 50)
    print("数据预处理...")
    n_before = len(df_ret)
    stats = {"原始记录": n_before}

    # 1. 合并上市日期
    df = df_ret.merge(df_company[["stkcd", "listdt", "stknme"]], on="stkcd", how="left")
    df["listdt"] = df["listdt"].fillna(pd.Timestamp("1990-01-01"))
    df["listed_months"] = (
        (df["trdmnt"].dt.year - df["listdt"].dt.year) * 12 +
        (df["trdmnt"].dt.month - df["listdt"].dt.month)
    )
    mask_new = df["listed_months"] < MIN_LISTED_MONTHS
    stats["上市不足6个月"] = mask_new.sum()
    df = df[~mask_new]

    # 2. 停牌处理：合并每月停牌天数
    if len(df_susp) > 0:
        df = df.merge(df_susp, on=["stkcd", "trdmnt"], how="left")
        df["susp_days_in_month"] = df["susp_days_in_month"].fillna(0)
    else:
        df["susp_days_in_month"] = 0
    mask_susp = df["susp_days_in_month"] > MIN_SUSP_DAYS
    stats["月内停牌>15天"] = mask_susp.sum()
    df = df[~mask_susp]

    # 3. ST股标记
    df["is_st"] = df["stknme"].str.contains(r"\*?ST", na=False, regex=True)
    mask_st = df["is_st"]
    stats["ST/*ST股"] = mask_st.sum()
    if ST_FLAG:
        df = df[~mask_st]

    # 4. 保留关键列
    df_clean = df[["stkcd", "trdmnt", "mretwd", "mretnd", "msmvosd",
                    "msmvttl", "ndaytrd", "markettype", "stknme",
                    "listed_months", "susp_days_in_month"]].copy()

    n_after = len(df_clean)
    stats["筛除后记录"] = n_after
    stats["剔除总数"] = n_before - n_after

    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"  保留率: {n_after/n_before*100:.1f}%")

    return df_clean


def preprocess_balance_sheet(df_bs):
    """
    取每只股票每年的年报（12月31日）账面权益
    """
    print("处理资产负债表（取年报）...")
    df_annual = df_bs[df_bs["accper"].dt.month == 12].copy()
    df_annual = df_annual.sort_values(["stkcd", "year"])
    df_annual = df_annual.drop_duplicates(subset=["stkcd", "year"], keep="last")
    df_annual = df_annual.rename(columns={"year": "fiscal_year"})
    print(f"  年报数据: {len(df_annual)} 条, {df_annual['stkcd'].nunique()} 只股票")
    return df_annual[["stkcd", "fiscal_year", "total_assets", "total_equity"]]


def preprocess_income_statement(df_is_):
    """
    取每只股票每年的年报（12月31日）归母净利润
    """
    print("处理利润表（取年报）...")
    df_annual = df_is_[df_is_["accper"].dt.month == 12].copy()
    df_annual = df_annual.sort_values(["stkcd", "year"])
    df_annual = df_annual.drop_duplicates(subset=["stkcd", "year"], keep="last")
    df_annual = df_annual.rename(columns={"year": "fiscal_year"})
    print(f"  年报数据: {len(df_annual)} 条, {df_annual['stkcd'].nunique()} 只股票")
    return df_annual[["stkcd", "fiscal_year", "net_profit_parent", "op_profit"]]
