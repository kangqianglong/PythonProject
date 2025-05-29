import pandas as pd
import numpy as np

#代码说明：
#数据生成：generate_sample_data 函数模拟生成了 100 只股票 252 天的收盘价、市值和市盈率数据。
#选股策略：stock_selection 函数根据市值和市盈率两个因子进行选股，选取市值前 20% 且市盈率后 20% 的股票。
#择时策略：timing_strategy 函数使用简单的移动平均线交叉策略，计算 5 日和 20 日移动平均线，当短期均线大于长期均线时发出买入信号，反之发出卖出信号。
#主函数：main 函数遍历所有交易日，依次执行选股和择时策略，并将信号存储在 all_signals_df 中。

# 模拟数据生成
def generate_sample_data():
    np.random.seed(0)
    num_stocks = 100
    num_days = 252
    dates = pd.date_range(start='2024-01-01', periods=num_days)
    stocks = [f'Stock_{i}' for i in range(num_stocks)]
    # 生成收盘价数据
    close_prices = pd.DataFrame(np.random.randn(num_days, num_stocks).cumsum(axis=0), index=dates, columns=stocks)
    # 生成市值数据
    market_cap = pd.DataFrame(np.random.randint(100, 1000, size=(num_days, num_stocks)), index=dates, columns=stocks)
    # 生成市盈率数据
    pe_ratio = pd.DataFrame(np.random.uniform(10, 50, size=(num_days, num_stocks)), index=dates, columns=stocks)
    return close_prices, market_cap, pe_ratio


# 选股函数
def stock_selection(market_cap, pe_ratio, date):
    # 选取市值前 20% 和市盈率后 20% 的股票
    market_cap_rank = market_cap.loc[date].rank(ascending=False)
    pe_ratio_rank = pe_ratio.loc[date].rank(ascending=True)
    selected_stocks = market_cap.columns[
        (market_cap_rank <= len(market_cap.columns) * 0.2) & (pe_ratio_rank <= len(pe_ratio.columns) * 0.2)]
    return selected_stocks


# 择时函数
def timing_strategy(close_prices, selected_stocks, date):
    signals = {}
    for stock in selected_stocks:
        short_ma = close_prices[stock].loc[:date].tail(5).mean()
        long_ma = close_prices[stock].loc[:date].tail(20).mean()
        if short_ma > long_ma:
            signals[stock] = 1  # 买入信号
        elif short_ma < long_ma:
            signals[stock] = -1  # 卖出信号
        else:
            signals[stock] = 0  # 持有信号
    return signals


# 主函数
def main():
    close_prices, market_cap, pe_ratio = generate_sample_data()
    trading_dates = close_prices.index
    all_signals = {}
    for date in trading_dates:
        selected_stocks = stock_selection(market_cap, pe_ratio, date)
        signals = timing_strategy(close_prices, selected_stocks, date)
        all_signals[date] = signals
    all_signals_df = pd.DataFrame(all_signals).T
    print(all_signals_df)


if __name__ == "__main__":
    main()
