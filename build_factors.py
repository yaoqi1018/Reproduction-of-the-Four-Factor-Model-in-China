"""
因子构建模块：MKT、SMB(调整后)、VMG(EP)、PMO(情绪) 四因子月收益率序列

中国四因子模型（CH4新版）：
  - MKT: 市场因子（全A股流通市值加权 - Rf）
  - SMB: 规模因子 = (EP中性SMB + 换手率中性SMB) / 2
  - VMG: 价值因子（高EP - 低EP），EP = 归母净利润 / 总市值
  - PMO: 情绪因子（低换手率 - 高换手率），基于异常换手率2x3分组
"""

import pandas as pd
import numpy as np
from config import *


def build_annual_ep(df_ret, df_is_):
    """
    按中国四因子标准方法构建每年6月底用于分组的 EP

    做法（与FF价值因子的时序逻辑一致）：
    - t年7月到t+1年6月，使用 t-1 年12月的年报数据
    - EP = net_profit_parent(t-1) / market_cap(Dec t-1)
    - 需确保净利润 > 0（亏损公司不纳入价值排序）
    """
    df = df_ret[["stkcd", "trdmnt", "msmvttl", "msmvosd", "mretwd"]].copy()
    df["month"] = df["trdmnt"].dt.month
    df["year"] = df["trdmnt"].dt.year

    # 每年12月的市值
    dec_mkt = df[df["month"] == 12][["stkcd", "year", "msmvttl"]].copy()
    dec_mkt = dec_mkt.rename(columns={"year": "fiscal_year", "msmvttl": "mkt_dec"})

    # 合并归母净利润 → EP
    dec_ep = dec_mkt.merge(df_is_, on=["stkcd", "fiscal_year"], how="inner")
    # 只保留正利润的股票（EP才有排序意义）
    dec_ep = dec_ep[dec_ep["net_profit_parent"] > 0]
    dec_ep["ep_annual"] = dec_ep["net_profit_parent"] / dec_ep["mkt_dec"]
    dec_ep = dec_ep[dec_ep["ep_annual"] > 0]

    # 分配：month>=7 → 用fiscal_year = year-1，否则 year-2
    df["ep_fy"] = np.where(df["month"] >= 7, df["year"] - 1, df["year"] - 2)
    df = df.merge(
        dec_ep[["stkcd", "fiscal_year", "ep_annual"]],
        left_on=["stkcd", "ep_fy"], right_on=["stkcd", "fiscal_year"], how="left"
    )

    df_valid = df.dropna(subset=["ep_annual"]).copy()
    # 每月缩尾1%
    for period in df_valid["trdmnt"].unique():
        mask = df_valid["trdmnt"] == period
        q01 = df_valid.loc[mask, "ep_annual"].quantile(0.01)
        q99 = df_valid.loc[mask, "ep_annual"].quantile(0.99)
        df_valid.loc[mask, "ep_annual"] = df_valid.loc[mask, "ep_annual"].clip(q01, q99)

    return df_valid.rename(columns={"ep_annual": "ep"})


def compute_market_factor(df, df_rf):
    """
    MKT = 全A股流通市值加权平均回报率 - 无风险利率
    """
    df_mkt = df.merge(df_rf, on="trdmnt", how="left")
    df_mkt = df_mkt.dropna(subset=["rf_monthly"])

    monthly_mkt = df_mkt.groupby("trdmnt")[["mretwd", "msmvosd"]].apply(
        lambda g: np.average(g["mretwd"], weights=g["msmvosd"]), include_groups=False
    ).reset_index(name="rm_vw")
    monthly_mkt = monthly_mkt.merge(df_rf, on="trdmnt", how="left")
    monthly_mkt["MKT"] = monthly_mkt["rm_vw"] - monthly_mkt["rf_monthly"]
    return monthly_mkt[["trdmnt", "rm_vw", "rf_monthly", "MKT"]]


def compute_smb_vmg(df):
    """
    2x3 分组：SMB 和 VMG

    每月：
    1. 规模：总市值中位数分 S / B
    2. 价值：EP 分 H(前30%) / M(中40%) / L(后30%)
    3. 6组合流通市值加权收益
    4. SMB = (SH+SM+SL)/3 - (BH+BM+BL)/3
    5. VMG = (SH+BH)/2 - (SL+BL)/2
    """
    df = df.dropna(subset=["msmvttl", "ep"]).copy()

    results = []
    months = sorted(df["trdmnt"].unique())

    for t in months:
        dt = df[df["trdmnt"] == t].copy()
        if len(dt) < 30:
            continue

        # 按市值分 S/B
        size_median = dt["msmvttl"].median()
        dt["size_group"] = np.where(dt["msmvttl"] >= size_median, "B", "S")

        # 按EP分 H/M/L（EP高=便宜=价值股）
        dt["ep_rank"] = dt["ep"].rank(pct=True)
        dt["ep_group"] = np.where(
            dt["ep_rank"] >= 1 - BM_HIGH_PCT, "H",
            np.where(dt["ep_rank"] < BM_LOW_PCT, "L", "M")
        )

        portfolios = {}
        for sg in ["S", "B"]:
            for bg in ["H", "M", "L"]:
                subset = dt[(dt["size_group"] == sg) & (dt["ep_group"] == bg)]
                if len(subset) == 0:
                    continue
                if USE_CIRCULATING_MV:
                    ret = np.average(subset["mretwd"], weights=subset["msmvosd"])
                else:
                    ret = subset["mretwd"].mean()
                portfolios[f"{sg}/{bg}"] = ret

        required = [f"{s}/{b}" for s in ["S", "B"] for b in ["H", "M", "L"]]
        if not all(k in portfolios for k in required):
            continue

        SMB = (portfolios["S/H"] + portfolios["S/M"] + portfolios["S/L"]) / 3 \
            - (portfolios["B/H"] + portfolios["B/M"] + portfolios["B/L"]) / 3
        VMG = (portfolios["S/H"] + portfolios["B/H"]) / 2 \
            - (portfolios["S/L"] + portfolios["B/L"]) / 2

        results.append({
            "trdmnt": t, "SMB": SMB, "VMG": VMG,
            "n_S": (dt["size_group"] == "S").sum(),
            "n_B": (dt["size_group"] == "B").sum(),
        })

    return pd.DataFrame(results)


def compute_abnormal_turnover(df_ret, df_turn):
    """
    计算异常换手率 = 本月日均换手率 / 过去12个月日均换手率均值
    需要至少6个月历史数据
    """
    df = df_ret[["stkcd", "trdmnt", "msmvttl", "msmvosd", "mretwd"]].copy()
    df = df.merge(df_turn[["stkcd", "trdmnt", "turn_daily_avg"]],
                   on=["stkcd", "trdmnt"], how="inner")

    df = df.sort_values(["stkcd", "trdmnt"])
    df["turn_avg_12m"] = df.groupby("stkcd")["turn_daily_avg"].transform(
        lambda x: x.rolling(12, min_periods=PM_MIN_TURN_MONTHS).mean()
    )

    df["abnormal_turn"] = df["turn_daily_avg"] / df["turn_avg_12m"]
    df = df.dropna(subset=["abnormal_turn"])
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["abnormal_turn"])

    return df


def compute_turnover_factors(df):
    """
    Size × Turnover 2x3 分组，产生：
      - PMO（情绪因子）= (S_low + B_low)/2 - (S_high + B_high)/2
      - SMB_turnover（换手率中性规模因子）

    步骤：
    1. 剔除市值最小30%壳股
    2. 按市值中位数分 S/B
    3. 按异常换手率分 Low(30%) / Mid(40%) / High(30%)
    4. 6个组合流通市值加权
    """
    df = df.dropna(subset=["msmvttl", "abnormal_turn"]).copy()

    results = []
    months = sorted(df["trdmnt"].unique())

    for t in months:
        dt = df[df["trdmnt"] == t].copy()
        if len(dt) < 30:
            continue

        # 剔除市值最小30%壳股
        size_p30 = dt["msmvttl"].quantile(PM_SHELL_PCT)
        dt = dt[dt["msmvttl"] >= size_p30].copy()
        if len(dt) < 20:
            continue

        # 按市值中位数分 S/B
        size_median = dt["msmvttl"].median()
        dt["size_group"] = np.where(dt["msmvttl"] >= size_median, "B", "S")

        # 按异常换手率分 Low/Mid/High
        dt["turn_rank"] = dt["abnormal_turn"].rank(pct=True)
        dt["turn_group"] = np.where(
            dt["turn_rank"] >= 1 - PM_HIGH_PCT, "H",
            np.where(dt["turn_rank"] < PM_LOW_PCT, "L", "M")
        )

        portfolios = {}
        for sg in ["S", "B"]:
            for tg in ["L", "M", "H"]:
                subset = dt[(dt["size_group"] == sg) & (dt["turn_group"] == tg)]
                if len(subset) == 0:
                    continue
                if USE_CIRCULATING_MV:
                    ret = np.average(subset["mretwd"], weights=subset["msmvosd"])
                else:
                    ret = subset["mretwd"].mean()
                portfolios[f"{sg}/{tg}"] = ret

        required = [f"{s}/{t}" for s in ["S", "B"] for t in ["L", "M", "H"]]
        if not all(k in portfolios for k in required):
            continue

        SMB_turnover = (portfolios["S/L"] + portfolios["S/M"] + portfolios["S/H"]) / 3 \
                     - (portfolios["B/L"] + portfolios["B/M"] + portfolios["B/H"]) / 3
        PMO = (portfolios["S/L"] + portfolios["B/L"]) / 2 \
            - (portfolios["S/H"] + portfolios["B/H"]) / 2

        results.append({
            "trdmnt": t, "SMB_turnover": SMB_turnover, "PMO": PMO,
        })

    return pd.DataFrame(results)


def build_all_factors(df_clean, df_rf, df_is_, df_turn):
    """构建四个因子：MKT, SMB(调整后), VMG, PMO(情绪)"""
    print("=" * 50)
    print("构建因子...")

    # 匹配 EP（归母净利润 / 市值）
    print("  匹配 EP 数据...")
    df_ep = build_annual_ep(df_clean, df_is_)
    print(f"  EP数据: {len(df_ep)} 条, 月数: {df_ep['trdmnt'].nunique()}")

    # 计算异常换手率
    print("  计算异常换手率...")
    df_turn_full = compute_abnormal_turnover(df_clean, df_turn)
    print(f"  异常换手率: {len(df_turn_full)} 条, 月数: {df_turn_full['trdmnt'].nunique()}")

    # MKT
    print("  计算 MKT...")
    df_mkt = compute_market_factor(df_clean, df_rf)
    print(f"    MKT: {len(df_mkt)} 个月, 均值={df_mkt['MKT'].mean():.4f}")

    # EP-based SMB & VMG（原有方法）
    print("  计算 SMB(EP中性) & VMG...")
    df_sv = compute_smb_vmg(df_ep)
    print(f"    SMB&VMG(EP): {len(df_sv)} 个月")

    # Turnover-based SMB & PMO（新方法）
    print("  计算 SMB(换手率中性) & PMO(情绪)...")
    df_st = compute_turnover_factors(df_turn_full)
    print(f"    SMB&PMO(换手率): {len(df_st)} 个月")

    # 合并
    factors = df_mkt.merge(df_sv, on="trdmnt", how="inner")
    factors = factors.merge(df_st, on="trdmnt", how="inner")

    # 调整后的 SMB = EP中性SMB 与 换手率中性SMB 的简单平均
    factors["SMB"] = (factors["SMB"] + factors["SMB_turnover"]) / 2

    factors = factors.sort_values("trdmnt").reset_index(drop=True)

    print(f"  最终因子: {len(factors)} 个月")
    print(f"  时间范围: {factors['trdmnt'].min().date()} ~ {factors['trdmnt'].max().date()}")
    return factors
