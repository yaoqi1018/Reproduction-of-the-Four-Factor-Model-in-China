"""
CSMAR数据加载模块
"""

import pandas as pd
import numpy as np
from config import *


def load_monthly_returns():
    """加载月个股回报率"""
    print("加载月个股回报率...")
    df = pd.read_csv(RETURN_FILE, dtype={"Stkcd": str})
    df = df.rename(columns={
        "Stkcd": "stkcd", "Trdmnt": "trdmnt",
        "Mretwd": "mretwd", "Mretnd": "mretnd",
        "Msmvosd": "msmvosd", "Msmvttl": "msmvttl",
        "Ndaytrd": "ndaytrd", "Markettype": "markettype",
        "Capchgdt": "capchgdt"
    })
    df["trdmnt"] = pd.to_datetime(df["trdmnt"].astype(str) + "-01")
    for col in ["mretwd", "mretnd", "msmvosd", "msmvttl"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["ndaytrd"] = pd.to_numeric(df["ndaytrd"], errors="coerce")

    # 基本筛除：无收益、无市值的数据
    df = df.dropna(subset=["mretwd", "msmvttl"])
    df = df[df["msmvttl"] > 0]

    # 市值单位换算
    df["msmvttl"] = df["msmvttl"] * MSMVTTL_SCALE
    df["msmvosd"] = df["msmvosd"] * MSMVTTL_SCALE

    df = df.sort_values(["trdmnt", "stkcd"]).reset_index(drop=True)
    n_stocks = df["stkcd"].nunique()
    n_months = df["trdmnt"].nunique()
    print(f"  个股回报率: {len(df)} 条, {n_stocks} 只股票, {n_months} 个月")
    return df


def load_riskfree():
    """加载无风险利率（月化），取每月第一条"""
    print("加载无风险利率...")
    df = pd.read_csv(RISKFREE_FILE, dtype={"Nrr1": str})
    df = df.rename(columns={
        "Nrr1": "nrr", "Clsdt": "trddt",
        "Nrrdata": "rf_annual", "Nrrdaydt": "rf_daily",
        "Nrrwkdt": "rf_weekly", "Nrrmtdt": "rf_monthly"
    })
    df["trdmnt"] = pd.to_datetime(df["trddt"]).dt.to_period("M").dt.to_timestamp()
    monthly = df.groupby("trdmnt")["rf_monthly"].first().reset_index()
    monthly["rf_monthly"] = pd.to_numeric(monthly["rf_monthly"], errors="coerce")
    # CSMAR无风险利率为百分比形式（0.2263=0.2263%），转为小数
    monthly["rf_monthly"] = monthly["rf_monthly"] / 100.0
    print(f"  无风险利率: {len(monthly)} 个月")
    return monthly


def load_balance_sheet():
    """加载资产负债表，提取股东权益（A003000000）"""
    print("加载资产负债表...")
    df = pd.read_csv(BALANCE_SHEET_FILE, dtype={"Stkcd": str, "Accper": str})
    df = df.rename(columns={
        "Stkcd": "stkcd", "Accper": "accper",
        "Typrep": "typrep", "IfCorrect": "if_correct",
        "A001000000": "total_assets",
        "A003000000": "total_equity"
    })
    df["accper"] = pd.to_datetime(df["accper"])
    # 保留合并报表 (A)，排除母公司报表
    df = df[df["typrep"] == "A"]
    for col in ["total_assets", "total_equity"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["total_equity"])
    df = df[df["total_equity"] > 0]
    df["year"] = df["accper"].dt.year
    print(f"  资产负债表: {len(df)} 条, {df['stkcd'].nunique()} 只股票")
    return df[["stkcd", "accper", "year", "total_assets", "total_equity"]]


def load_income_statement():
    """加载利润表，提取归母净利润（B002000101）和营业利润（B001100000）"""
    print("加载利润表...")
    df = pd.read_csv(INCOME_STATEMENT_FILE, dtype={"Stkcd": str, "Accper": str})
    df = df.rename(columns={
        "Stkcd": "stkcd", "Accper": "accper",
        "Typrep": "typrep",
        "B001100000": "op_profit",
        "B002000101": "net_profit_parent",
        "B001200000": "total_profit",
    })
    df["accper"] = pd.to_datetime(df["accper"])
    df = df[df["typrep"] == "A"]  # 合并报表
    for col in ["op_profit", "net_profit_parent"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["net_profit_parent"])
    df["year"] = df["accper"].dt.year
    print(f"  利润表: {len(df)} 条, {df['stkcd'].nunique()} 只股票")
    return df[["stkcd", "accper", "year", "op_profit", "net_profit_parent"]]


def load_company_info():
    """加载公司基本情况表"""
    print("加载公司基本情况表...")
    df = pd.read_csv(COMPANY_INFO_FILE, dtype={"Stkcd": str, "Listdt": str})
    df = df.rename(columns={
        "Stkcd": "stkcd", "Stknme": "stknme",
        "Listdt": "listdt", "Indcd": "indcd",
        "Indnme": "indnme", "Listexg": "listexg"
    })
    df["listdt"] = pd.to_datetime(df["listdt"], format="%Y-%m-%d", errors="coerce")
    df = df.dropna(subset=["listdt"])
    print(f"  公司基本情况: {len(df)} 只股票")
    return df[["stkcd", "stknme", "listdt", "indcd", "indnme", "listexg"]]


def load_suspension():
    """加载停复牌信息，统计每只股票每月的停牌天数"""
    print("加载停复牌信息...")
    df = pd.read_csv(SUSPENSION_FILE, dtype={"Stkcd": str})
    df = df.rename(columns={
        "Stkcd": "stkcd", "Stknme": "stknme",
        "Suspdate": "susp_date", "Resmdate": "resm_date",
        "Timeperd": "susp_days"
    })
    df["susp_date"] = pd.to_datetime(df["susp_date"], format="%Y-%m-%d", errors="coerce")
    df["resm_date"] = pd.to_datetime(df["resm_date"], format="%Y-%m-%d", errors="coerce")
    df["susp_days"] = pd.to_numeric(df["susp_days"], errors="coerce")

    # 统计每只股票每月的停牌天数
    records = []
    for _, row in df.iterrows():
        if pd.isna(row["susp_date"]) or pd.isna(row["resm_date"]):
            continue
        # 跨越的月份
        cur = row["susp_date"].replace(day=1)
        end = row["resm_date"].replace(day=1)
        while cur <= end:
            ym = cur
            # 计算该月内停牌天数
            month_start = ym
            month_end = ym + pd.offsets.MonthEnd(1)
            s = max(row["susp_date"], month_start)
            e = min(row["resm_date"], month_end)
            if s <= e:
                days = (e - s).days + 1
                records.append({"stkcd": row["stkcd"], "trdmnt": ym, "susp_days_in_month": days})
            cur += pd.offsets.MonthBegin(1)

    if not records:
        print("  停牌信息: 无需处理或格式异常, 返回空")
        return pd.DataFrame(columns=["stkcd", "trdmnt", "susp_days_in_month"])

    df_monthly = pd.DataFrame(records)
    df_monthly = df_monthly.groupby(["stkcd", "trdmnt"])["susp_days_in_month"].sum().reset_index()
    print(f"  停牌信息: {len(df_monthly)} 条月度停牌记录")
    return df_monthly


def load_official_factors():
    """加载官方Carhart四因子数据（CH4月度数据）"""
    print("加载官方四因子数据...")
    df = pd.read_excel(OFFICIAL_FACTOR_FILE)
    df = df.rename(columns={
        "mnthdt": "mnthdt",
        "mktrf": "MKT",
        "SMB": "SMB",
        "VMG": "VMG",
        "PMO": "PMO",
    })
    df["trdmnt"] = pd.to_datetime(df["mnthdt"].astype(str), format="%Y%m%d")
    df["trdmnt"] = df["trdmnt"].dt.to_period("M").dt.to_timestamp()
    for c in ["MKT", "SMB", "VMG", "PMO"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    print(f"  官方因子: {len(df)} 个月")
    return df[["trdmnt", "MKT", "SMB", "VMG", "PMO"]]


def load_turnover():
    """加载月个股换手率"""
    print("加载月个股换手率...")
    df = pd.read_csv(TURNOVER_FILE, dtype={"Stkcd": str})
    df = df.rename(columns={
        "Stkcd": "stkcd", "Trdmnt": "trdmnt",
        "ToverOsMAvg": "turn_daily_avg",
    })
    df["trdmnt"] = pd.to_datetime(df["trdmnt"].astype(str) + "-01")
    df["turn_daily_avg"] = pd.to_numeric(df["turn_daily_avg"], errors="coerce")
    df = df.dropna(subset=["turn_daily_avg"])
    df = df[df["turn_daily_avg"] > 0]
    df = df.sort_values(["stkcd", "trdmnt"]).reset_index(drop=True)
    print(f"  换手率: {len(df)} 条, {df['stkcd'].nunique()} 只股票")
    return df[["stkcd", "trdmnt", "turn_daily_avg"]]
