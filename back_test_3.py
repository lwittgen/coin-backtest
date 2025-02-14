import pandas as pd
import numpy as np
from datetime import datetime
import itertools


def run_backtest(
    df, short_period, long_period, initial_balance=1000000, commission_pct=0.01
):
    """
    이동평균선 돌파 전략 백테스트 실행

    Parameters:
    - df: DataFrame (datetime, trade_price 컬럼 포함)
    - short_period: 단기 이동평균 기간
    - long_period: 장기 이동평균 기간
    - initial_balance: 초기 잔고
    - commission_pct: 거래 수수료 (%)
    """
    # 이동평균 계산
    df["MA_short"] = df["trade_price"].rolling(window=short_period).mean()
    df["MA_long"] = df["trade_price"].rolling(window=long_period).mean()

    # 시그널 생성
    df["prev_short_above_long"] = (
        df["MA_short"].shift(1) > df["MA_long"].shift(1)
    ).astype(int)
    df["curr_short_above_long"] = (df["MA_short"] > df["MA_long"]).astype(int)

    df["buy_signal"] = (df["prev_short_above_long"] == 0) & (
        df["curr_short_above_long"] == 1
    )
    df["sell_signal"] = (df["prev_short_above_long"] == 1) & (
        df["curr_short_above_long"] == 0
    )

    # 백테스트 실행
    balance = initial_balance
    position = 0
    trades = []
    yearly_results = {}

    for idx, row in df.iterrows():
        year = pd.to_datetime(row["datetime"]).year

        if row["buy_signal"] and balance > 0:
            # 매수 실행
            position = balance * (1 - commission_pct) / row["trade_price"]
            balance = 0
            trades.append(
                {
                    "datetime": row["datetime"],
                    "type": "buy",
                    "price": row["trade_price"],
                    "position": position,
                    "balance": balance,
                }
            )

        elif row["sell_signal"] and position > 0:
            # 매도 실행
            balance = position * row["trade_price"] * (1 - commission_pct)
            position = 0
            trades.append(
                {
                    "datetime": row["datetime"],
                    "type": "sell",
                    "price": row["trade_price"],
                    "position": position,
                    "balance": balance,
                }
            )

        # 연말 결과 저장
        if (
            idx == len(df) - 1
            or pd.to_datetime(df.iloc[idx + 1]["datetime"]).year != year
        ):
            current_value = balance + (
                position * row["trade_price"] if position > 0 else 0
            )
            yearly_results[year] = current_value

    # 최종 가치 계산
    final_value = balance + (
        position * df.iloc[-1]["trade_price"] if position > 0 else 0
    )

    return trades, yearly_results, final_value


def backtest_all_parameters(csv_file, output_file):
    """
    모든 파라미터 조합에 대해 백테스트 실행

    Parameters:
    - csv_file: 입력 CSV 파일 경로
    - output_file: 결과 저장할 CSV 파일 경로
    """
    # CSV 파일 로드
    df = pd.read_csv(csv_file)
    df["datetime"] = pd.to_datetime(df["datetime"])

    # 파라미터 조합 생성
    short_periods = range(5, 21)  # 5-20
    long_periods = range(20, 81)  # 20-80
    results = []

    for short_period, long_period in itertools.product(short_periods, long_periods):
        if short_period >= long_period:
            continue

        trades, yearly_results, final_value = run_backtest(
            df, short_period, long_period
        )

        # 연간 수익률 계산
        prev_value = 1000000  # 초기 잔고
        yearly_returns = {}
        for year, value in yearly_results.items():
            yearly_return = (value - prev_value) / prev_value * 100
            yearly_returns[year] = yearly_return
            prev_value = value

        # 전체 수익률 계산
        total_return = (final_value - 1000000) / 1000000 * 100

        results.append(
            {
                "short_period": short_period,
                "long_period": long_period,
                "final_value": final_value,
                "total_return": total_return,
                **{f"value_{year}": value for year, value in yearly_results.items()},
                **{f"return_{year}": ret for year, ret in yearly_returns.items()},
            }
        )

    # 결과를 DataFrame으로 변환하고 CSV로 저장
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)

    return results_df


if __name__ == "__main__":
    # 사용 예시
    input_csv = "candles_days.csv"  # 입력 데이터 파일명
    output_csv = "backtest_results.csv"  # 결과 저장할 파일명

    results = backtest_all_parameters(input_csv, output_csv)

    # 최고 수익률 결과 출력
    best_result = results.loc[results["total_return"].idxmax()]
    print(f"\n최고 수익률 결과:")
    print(f"단기이평: {best_result['short_period']}")
    print(f"장기이평: {best_result['long_period']}")
    print(f"최종잔고: {best_result['final_value']:,.0f}원")
    print(f"전체수익률: {best_result['total_return']:.2f}%")
