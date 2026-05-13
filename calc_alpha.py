
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from config import *

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


def load_csi300_monthly_from_daily(filepath):
    """从日指数数据计算CSI300月回报率"""
    df = pd.read_csv(filepath, dtype={"Indexcd": str, "Trddt": str})
    df = df.rename(columns={
        "Indexcd": "indexcd", "Trddt": "trddt",
        "Retindex": "ret_daily"
    })
    df = df[df["indexcd"] == "000300"].copy()
    df["trddt"] = pd.to_datetime(df["trddt"], format="%Y-%m-%d")
    df["ret_daily"] = pd.to_numeric(df["ret_daily"], errors="coerce")
    df = df.dropna(subset=["ret_daily"])

    # 按月复利
    df["trdmnt"] = df["trddt"].dt.to_period("M").dt.to_timestamp()
    monthly = df.groupby("trdmnt")["ret_daily"].apply(
        lambda x: np.prod(1 + x) - 1
    ).reset_index(name="mret_csi300")

    print(f"  CSI300月回报率: {len(monthly)} 个月")
    print(f"  日期范围: {monthly['trdmnt'].min().date()} ~ {monthly['trdmnt'].max().date()}")
    return monthly


def compute_alpha(df_csi300, factors):
    """
    R_CSI300 - Rf = α + β1·MKT + β2·SMB + β3·HML + β4·PMO + ε
    factors 已含 rf_monthly，直接计算超额收益
    """
    df = df_csi300.merge(factors, on="trdmnt", how="inner")
    df["excess"] = df["mret_csi300"] - df["rf_monthly"]

    factor_cols = ["MKT", "SMB", "VMG", "PMO"]
    df = df.dropna(subset=["excess"] + factor_cols)

    y = df["excess"].values
    X = df[factor_cols].values
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()

    return model, df


def plot_alpha_analysis(df, model):
    """绘制alpha分析图"""
    factor_cols = ["MKT", "SMB", "VMG", "PMO"]

    # 图1：CSI300超额收益 vs 拟合值
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 实际 vs 拟合
    ax = axes[0, 0]
    fitted = model.fittedvalues
    actual = df["excess"].values
    ax.scatter(fitted, actual, alpha=0.4, s=15, edgecolors="none")
    lims = [min(fitted.min(), actual.min()), max(fitted.max(), actual.max())]
    ax.plot(lims, lims, "r--", linewidth=0.8)
    ax.set_xlabel("因子模型拟合值")
    ax.set_ylabel("CSI300实际超额收益")
    ax.set_title(f"Actual vs Fitted (R-sq={model.rsquared:.3f})")

    # 时间序列：累计超额收益
    ax = axes[0, 1]
    cum_excess = (1 + df["excess"]).cumprod()
    cum_fitted = (1 + pd.Series(fitted, index=df.index)).cumprod()
    ax.plot(df["trdmnt"], cum_excess, linewidth=1, label="CSI300累计超额收益")
    ax.plot(df["trdmnt"], cum_fitted, linewidth=1, alpha=0.7, label="因子模型累计拟合")
    ax.axhline(y=1, color="grey", linestyle="--", linewidth=0.5)
    ax.legend(fontsize=8)
    ax.set_title("累计超额收益 vs 因子模型")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # 残差时间序列
    ax = axes[1, 0]
    residuals = model.resid
    ax.plot(df["trdmnt"], residuals, linewidth=0.5, alpha=0.7)
    ax.axhline(y=0, color="grey", linestyle="--", linewidth=0.5)
    ax.set_title("回归残差")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # 因子载荷柱状图
    ax = axes[1, 1]
    betas = model.params[1:]
    beta_names = factor_cols
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    bars = ax.bar(beta_names, betas, color=colors, alpha=0.8)
    ax.axhline(y=0, color="grey", linewidth=0.5)
    ax.set_title("Factor Loadings")
    for bar, val in zip(bars, betas):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (0.02 if val >= 0 else -0.04),
                f"{val:.3f}", ha="center", fontsize=10)

    plt.suptitle("沪深300 四因子模型 Alpha 分析", fontsize=14)
    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(f"{OUTPUT_DIR}/csi300_alpha_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  分析图已保存: {OUTPUT_DIR}/csi300_alpha_analysis.png")


def print_results(model, n_periods):
    """输出结果"""
    print("\n" + "=" * 60)
    print("沪深300 四因子模型回归结果")
    print("=" * 60)

    alpha = model.params[0]
    print(f"\n  Alpha (月化):  {alpha:.6f}")
    print(f"  Alpha (年化):  {alpha * 12:.6f}")
    print(f"  Alpha t值:    {model.tvalues[0]:.4f}")
    print(f"  Alpha p值:    {model.pvalues[0]:.4f}")
    print(f"  R-squared:    {model.rsquared:.4f}")
    print(f"  Adj R-sq:     {model.rsquared_adj:.4f}")
    print(f"  样本数:       {n_periods}")

    print(f"\n  因子载荷:")
    names = ["MKT", "SMB", "VMG", "PMO"]
    for i, name in enumerate(names, 1):
        b = model.params[i]
        t = model.tvalues[i]
        p = model.pvalues[i]
        stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
        print(f"    beta_{name}: {b:+.4f}  (t={t:+7.3f}, p={p:.4f}) {stars}")

    # F检验
    print(f"\n  F统计量:      {model.fvalue:.4f}")
    print(f"  F p值:        {model.f_pvalue:.6f}")


def main():
    print("=" * 60)
    print("任务3：沪深300因子调整 Alpha")
    print("=" * 60)

    # 加载因子（已含rf_monthly）
    factors = pd.read_csv(FACTOR_OUTPUT)
    factors["trdmnt"] = pd.to_datetime(factors["trdmnt"])
    print(f"已加载因子: {len(factors)} 个月")

    # 加载CSI300日数据 → 算月回报率
    df_csi300 = load_csi300_monthly_from_daily(f"{DATA_DIR}/指数信息.csv")

    # 回归
    print("\n运行因子回归...")
    model, df_reg = compute_alpha(df_csi300, factors)

    # 输出和可视化
    print_results(model, len(df_reg))
    plot_alpha_analysis(df_reg, model)

    # 保存回归结果
    result_df = pd.DataFrame({
        "trdmnt": df_reg["trdmnt"],
        "excess_csi300": df_reg["excess"],
        "fitted": model.fittedvalues,
        "residual": model.resid,
    })
    result_df.to_csv(f"{OUTPUT_DIR}/csi300_alpha_regression.csv",
                     index=False, encoding="utf-8-sig")
    print(f"\n回归结果已保存: {OUTPUT_DIR}/csi300_alpha_regression.csv")

    return model, df_reg


if __name__ == "__main__":
    main()
