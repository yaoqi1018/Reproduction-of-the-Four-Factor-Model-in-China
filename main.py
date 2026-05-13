"""
中国四因子模型 CH4 主流程
任务2：构建四因子 + 与官方因子对比 + 可视化
"""

import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from config import *
from load_data import (
    load_monthly_returns, load_riskfree, load_balance_sheet,
    load_company_info, load_suspension, load_official_factors,
    load_income_statement, load_turnover,
)
from preprocess import preprocess_returns, preprocess_income_statement
from build_factors import build_all_factors


# 中文字体设置
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei"]
plt.rcParams["axes.unicode_minus"] = False


def compare_with_official(my_factors, official):
    """对比自建因子与官方因子"""
    print("\n" + "=" * 60)
    print("与官方因子对比")
    print("=" * 60)

    official_renamed = official.rename(columns={
        "MKT": "MKT_o", "SMB": "SMB_o", "VMG": "VMG_o", "PMO": "PMO_o"
    })
    merged = my_factors[["trdmnt", "MKT", "SMB", "VMG", "PMO"]].merge(
        official_renamed, on="trdmnt", how="inner"
    )
    print(f"  重叠月份: {len(merged)}")

    # 我的因子 vs 官方因子对应关系
    pairs = [
        ("MKT", "MKT_o", "市场因子 MKT"),
        ("SMB", "SMB_o", "规模因子 SMB"),
        ("VMG", "VMG_o", "价值因子 VMG"),
        ("PMO", "PMO_o", "情绪因子 PMO"),
    ]

    print("\n因子相关性:")
    corr_data = {}
    for my, off, name in pairs:
        valid = merged[[my, off]].dropna()
        r = valid[my].corr(valid[off])
        print(f"  {name}: r = {r:.4f}  (n={len(valid)})")
        corr_data[name] = r

    return merged, pairs, corr_data


def plot_factor_comparison(merged, pairs):
    """绘制自建因子 vs 官方因子对比图"""
    print("\n绘制因子对比图...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for i, (my, off, name) in enumerate(pairs):
        ax = axes[i]
        valid = merged[[my, off, "trdmnt"]].dropna()
        ax.plot(valid["trdmnt"], valid[my], alpha=0.7, linewidth=0.8, label="自建因子")
        ax.plot(valid["trdmnt"], valid[off], alpha=0.7, linewidth=0.8, label="官方因子")
        r = valid[my].corr(valid[off])
        ax.set_title(f"{name}  (r={r:.4f})", fontsize=12)
        ax.axhline(y=0, color="grey", linestyle="--", linewidth=0.5)
        ax.legend(fontsize=8)
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(axis="x", rotation=45)

    plt.suptitle("自建因子 vs 官方因子（综合A股 流通市值加权）", fontsize=14)
    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/factor_comparison_ts.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  时序图已保存: {OUTPUT_DIR}/factor_comparison_ts.png")


def plot_scatter_comparison(merged, pairs):
    """绘制散点图 + 回归线"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()

    for i, (my, off, name) in enumerate(pairs):
        ax = axes[i]
        valid = merged[[my, off]].dropna()
        ax.scatter(valid[off], valid[my], alpha=0.3, s=10, edgecolors="none")
        # 45度线
        lims = [min(valid[off].min(), valid[my].min()),
                 max(valid[off].max(), valid[my].max())]
        ax.plot(lims, lims, "r--", linewidth=0.8, alpha=0.5)
        # 拟合线
        slope, intercept = np.polyfit(valid[off], valid[my], 1)
        ax.plot(valid[off], slope * valid[off] + intercept, "b-", linewidth=1, alpha=0.5)
        r = valid[my].corr(valid[off])
        ax.set_xlabel("官方因子")
        ax.set_ylabel("自建因子")
        ax.set_title(f"{name}\nr={r:.4f}, β={slope:.3f}", fontsize=11)

    plt.suptitle("自建因子 vs 官方因子 散点图", fontsize=14)
    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/factor_comparison_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  散点图已保存: {OUTPUT_DIR}/factor_comparison_scatter.png")


def plot_cumulative_returns(factors):
    """绘制四因子累计收益"""
    fig, ax = plt.subplots(figsize=(14, 5))
    factor_cols = ["MKT", "SMB", "VMG", "PMO"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for col, c in zip(factor_cols, colors):
        cum = (1 + factors[col]).cumprod()
        ax.plot(factors["trdmnt"], cum, linewidth=1, color=c, label=col)

    ax.axhline(y=1, color="grey", linestyle="--", linewidth=0.5)
    ax.legend()
    ax.set_title("四因子累计收益")
    ax.set_ylabel("累计净值")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/factor_cumulative.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  累计收益图已保存: {OUTPUT_DIR}/factor_cumulative.png")


def plot_correlation_heatmap(merged, pairs):
    """因子间相关性热力图"""
    factor_cols = ["MKT", "SMB", "VMG", "PMO"]
    corr = merged[factor_cols].corr()

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(factor_cols)))
    ax.set_yticks(range(len(factor_cols)))
    ax.set_xticklabels(factor_cols, fontsize=11)
    ax.set_yticklabels(factor_cols, fontsize=11)
    for i in range(len(factor_cols)):
        for j in range(len(factor_cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                    fontsize=12, color="white" if abs(corr.iloc[i, j]) > 0.5 else "black")
    ax.set_title("自建四因子相关性矩阵", fontsize=13)
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/factor_correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  相关性热力图已保存: {OUTPUT_DIR}/factor_correlation_heatmap.png")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ============ 第一步：加载数据 ============
    print("=" * 60)
    print("第一步：加载数据")
    print("=" * 60)
    df_ret = load_monthly_returns()
    df_rf = load_riskfree()
    df_is_ = load_income_statement()
    df_company = load_company_info()
    df_susp = load_suspension()
    df_turn = load_turnover()

    # ============ 第二步：预处理 ============
    print("\n" + "=" * 60)
    print("第二步：数据预处理")
    print("=" * 60)
    df_clean = preprocess_returns(df_ret, df_company, df_susp)
    df_is_annual = preprocess_income_statement(df_is_)

    # ============ 第三步：构建因子 ============
    print("\n" + "=" * 60)
    print("第三步：构建四因子")
    print("=" * 60)
    factors = build_all_factors(df_clean, df_rf, df_is_annual, df_turn)

    # ============ 第四步：因子统计 ============
    print("\n" + "=" * 60)
    print("第四步：因子描述性统计")
    print("=" * 60)
    factor_cols = ["MKT", "SMB", "VMG", "PMO"]
    n = len(factors)
    stats_rows = []
    for col in factor_cols:
        mu = factors[col].mean()
        sd = factors[col].std()
        t_stat = mu / (sd / np.sqrt(n)) if sd > 0 else 0
        stats_rows.append({
            "因子": col, "月均值": mu, "月标准差": sd,
            "年化均值": mu * 12, "年化标准差": sd * np.sqrt(12),
            "t值": t_stat, "最小": factors[col].min(), "最大": factors[col].max(),
        })
    stats_df = pd.DataFrame(stats_rows)
    print(stats_df.to_string(index=False))

    # 相关性矩阵
    print("\n因子间相关性:")
    print(factors[factor_cols].corr().to_string())

    # ============ 第五步：与官方因子对比 ============
    print("\n" + "=" * 60)
    print("第五步：加载官方因子并对比")
    print("=" * 60)
    official = load_official_factors()
    merged, pairs, corr_data = compare_with_official(factors, official)

    # ============ 第六步：可视化 ============
    print("\n" + "=" * 60)
    print("第六步：可视化")
    print("=" * 60)

    plot_factor_comparison(merged, pairs)
    plot_scatter_comparison(merged, pairs)
    plot_cumulative_returns(factors)
    plot_correlation_heatmap(merged, pairs)

    # ============ 保存结果 ============
    factors.to_csv(FACTOR_OUTPUT, index=False, encoding="utf-8-sig")
    stats_df.to_csv(FACTOR_STATS, index=False, encoding="utf-8-sig")
    print(f"\n因子数据已保存: {FACTOR_OUTPUT}")
    print(f"因子统计已保存: {FACTOR_STATS}")
    print("\n全部完成！")

    return factors


if __name__ == "__main__":
    factors = main()
