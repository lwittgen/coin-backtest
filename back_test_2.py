import pandas as pd
import numpy as np
from datetime import datetime
import csv


class MAStrategy:
    def __init__(self, initial_balance=1000000, commission_rate=0.0015):
        self.initial_balance = initial_balance
        self.commission_rate = commission_rate

    def calculate_ma(self, data, n, m):
        """Calculate moving averages"""
        df = data.copy()
        df["MA_short"] = df["trade_price"].rolling(window=n).mean()
        df["MA_long"] = df["trade_price"].rolling(window=m).mean()
        return df

    def simulate(self, data, n, m):
        """Run simulation for given parameters"""
        df = self.calculate_ma(data, n, m)

        balance = self.initial_balance
        position = 0  # 0: no position, 1: long position
        trade_history = []

        # Skip initial rows where MA is not available
        start_idx = max(n, m)

        for i in range(start_idx, len(df) - 1):
            current_price = df.iloc[i]["trade_price"]
            next_price = df.iloc[i + 1]["trade_price"]
            current_short_ma = df.iloc[i]["MA_short"]
            current_long_ma = df.iloc[i]["MA_long"]

            # Buy signal
            if (
                position == 0
                and current_short_ma < current_long_ma
                and next_price >= current_long_ma
            ):
                position = 1
                shares = balance / (next_price * (1 + self.commission_rate))
                cost = shares * next_price * (1 + self.commission_rate)
                balance -= cost
                trade_history.append(
                    {
                        "datetime": df.iloc[i + 1]["datetime"],
                        "action": "buy",
                        "price": next_price,
                        "shares": shares,
                        "balance": balance + shares * next_price,
                        "type": "entry",
                    }
                )

            # Sell signal
            elif (
                position == 1
                and current_short_ma > current_long_ma
                and next_price <= current_long_ma
            ):
                proceeds = shares * next_price * (1 - self.commission_rate)
                balance += proceeds
                position = 0
                trade_history.append(
                    {
                        "datetime": df.iloc[i + 1]["datetime"],
                        "action": "sell",
                        "price": next_price,
                        "shares": shares,
                        "balance": balance,
                        "type": "exit",
                    }
                )

        # Force close any remaining position at the end
        if position == 1:
            final_price = df.iloc[-1]["trade_price"]
            proceeds = shares * final_price * (1 - self.commission_rate)
            balance += proceeds
            trade_history.append(
                {
                    "datetime": df.iloc[-1]["datetime"],
                    "action": "sell",
                    "price": final_price,
                    "shares": shares,
                    "balance": balance,
                    "type": "final_exit",
                }
            )

        return pd.DataFrame(trade_history), balance

    def run_backtest(self, data, n_range, m_range):
        """Run backtest for different MA combinations"""
        results = []

        for n in range(n_range[0], n_range[1] + 1):
            for m in range(m_range[0], m_range[1] + 1):
                trades, final_balance = self.simulate(data, n, m)

                if len(trades) == 0:
                    continue

                # Calculate yearly returns
                trades["year"] = pd.to_datetime(trades["datetime"]).dt.year
                yearly_results = []

                years = trades["year"].unique()
                prev_year_balance = self.initial_balance

                for year in years:
                    year_end_trade = trades[trades["year"] == year].iloc[-1]
                    year_end_balance = year_end_trade["balance"]
                    yearly_return = (year_end_balance / prev_year_balance - 1) * 100
                    yearly_results.append(
                        {
                            "year": year,
                            "balance": year_end_balance,
                            "return": yearly_return,
                        }
                    )
                    prev_year_balance = year_end_balance

                total_return = (final_balance / self.initial_balance - 1) * 100

                results.append(
                    {
                        "n": n,
                        "m": m,
                        "final_balance": final_balance,
                        "total_return": total_return,
                        "yearly_results": yearly_results,
                        "trade_count": len(trades),
                    }
                )

        return results


def save_results(results, output_file):
    """Save backtest results to CSV"""
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(
            ["n", "m", "final_balance", "total_return", "trade_count", "yearly_details"]
        )

        for result in results:
            yearly_details = "; ".join(
                [
                    f"{y['year']}: ₩{y['balance']:,.0f} ({y['return']:.2f}%)"
                    for y in result["yearly_results"]
                ]
            )

            writer.writerow(
                [
                    result["n"],
                    result["m"],
                    f"₩{result['final_balance']:,.0f}",
                    f"{result['total_return']:.2f}%",
                    result["trade_count"],
                    yearly_details,
                ]
            )


def main():
    # Configuration
    INITIAL_BALANCE = 1000000  # 1백만원
    COMMISSION_RATE = 0.01
    N_MIN = 5
    N_MAX = 20
    M_MIN = 20
    M_MAX = 80

    # Read data
    data = pd.read_csv("candles_days.csv")
    data["datetime"] = pd.to_datetime(data["datetime"])

    # Initialize strategy
    strategy = MAStrategy(INITIAL_BALANCE, COMMISSION_RATE)

    # Run backtest
    results = strategy.run_backtest(
        data, n_range=(N_MIN, N_MAX), m_range=(M_MIN, M_MAX)
    )

    # Save results
    save_results(results, "backtest_2_days.csv")

    # Print summary of best performing strategy
    if results:
        best_result = max(results, key=lambda x: x["final_balance"])
        print(f"\nBest Performance:")
        print(f"Short MA Period (n): {best_result['n']}")
        print(f"Long MA Period (m): {best_result['m']}")
        print(f"Final Balance: ₩{best_result['final_balance']:,.0f}")
        print(f"Total Return: {best_result['total_return']:.2f}%")
        print(f"Number of Trades: {best_result['trade_count']}")
        print("\nYearly Performance:")
        for year_result in best_result["yearly_results"]:
            print(
                f"{year_result['year']}: ₩{year_result['balance']:,.0f} ({year_result['return']:.2f}%)"
            )


if __name__ == "__main__":
    main()
