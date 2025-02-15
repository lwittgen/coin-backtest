import pandas as pd
import numpy as np
from datetime import datetime
import csv


# 원본 데이터: csv 파일.
# 각 줄마다 캔들 정보가 있음.
# datetime 컬럼은 캔들의 시작 시각.
# trade_price 컬럼은 캔들 종료 시점의 가격.
# csv 파일에서 필요한 컬럼만 로드해서 사용.
# 제일 과거 데이터부터 시간순으로 저장이 되어 있음.

# 목표: 원본 파일에 있는 과거 데이터 기준으로 투자 시뮬레이션을 진행하고, 그 결과를 출력.
# 단기 이동평균선 (n개 이동평균)과 장기 이동평균선 (m개 이동평균), 그리고 rsi와 볼린저 밴드를 계산해야 함.
# n은 2개부터 20개까지, m은 4개부터 60개까지 변화시키면서 투자 시뮬레이션을 진행해야 함. 단, m이 n의 2배 미만인 경우는 제외.
# 단기 이평선이 장기 이평선 미만이고, rsi와 볼린저 밴드가 각각 자신의 하단선 미만인 경우에 매수함.
# 단기 이평선이 장기 이평선 이상, rsi와 볼린저 밴드가 각각 자신의 상단선 이상인 3가지 조건중 2개 이상 만족할 때 매도.
# 매매를 할때마다 매매 금액의 일정 비율로 수수료가 발생하는 데, 기본값은 1.0%로 지정.
# 매수할때는 가진 현금으로 (수수료를 포함해) 가능한 최대 양만큼 매수, 매도할때는 전액 매도.
# 매수 수량은 자연수가 아니라 소수도 가능.

# 출력 파일: csv 파일.
# 각 줄마다 주어진 n과 m에 대한 시뮬레이션 결과를 저장.
# 최종 잔고, 최종 수익률, 매매수, 수익이 난 해의 수, 손실난 해의 수, 각 연도별 수익률을 저장.
# 출려되는 잔고는 보유 현금과 보유 자산의 평가 금액을 합산한 금액.
# 연도별 수익률은 전년 마지막 날의 잔고 (현금 + 자산 평가 금액) 대비 올해 마지막 날의 잔고의 수익률.
# 최종 잔고 역시 현금과 평가 금액의 합산이며, 최종 수익률은 최종 잔고 대비 초기 자본금의 수익률.

# 화면 출력:
# 주어진 n과 m에 대한 시뮬레이션이 끝날때마다 한줄로 짧게 요약해서 출력. (n, m, 최종 잔고, 수익이 난 해의 수, 손실이 난 해의 수)
# 최고의 수익률을 기록한 n과 m및 최종 잔고와 수익률을 출력.


class InvestmentSimulator:
    def __init__(self, initial_capital=1_000_000, commission_rate=0.01):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate

    def calculate_indicators(self, df, short_period, long_period):
        # 이동평균선 계산
        df["short_ma"] = df["trade_price"].rolling(window=short_period).mean()
        df["long_ma"] = df["trade_price"].rolling(window=long_period).mean()

        # RSI 계산
        delta = df["trade_price"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))

        # 볼린저 밴드 계산
        df["bb_middle"] = df["trade_price"].rolling(window=20).mean()
        bb_std = df["trade_price"].rolling(window=20).std()
        df["bb_upper"] = df["bb_middle"] + 2 * bb_std
        df["bb_lower"] = df["bb_middle"] - 2 * bb_std

        return df

    def simulate(self, data_path, n, m):
        # 데이터 로드
        df = pd.read_csv(data_path, usecols=["datetime", "trade_price"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = self.calculate_indicators(df, n, m)

        # 초기화
        cash = self.initial_capital
        holdings = 0
        trades = 0
        yearly_returns = {}

        # 매매 시뮬레이션
        for i in range(max(m, 20), len(df)):
            current_price = df.iloc[i]["trade_price"]

            # 매수 조건 체크
            if (
                holdings == 0
                and df.iloc[i]["short_ma"] < df.iloc[i]["long_ma"]
                and df.iloc[i]["rsi"] < 30
                and df.iloc[i]["trade_price"] < df.iloc[i]["bb_lower"]
            ):

                # 수수료를 고려한 최대 매수 가능 수량
                holdings = (cash / (1 + self.commission_rate)) / current_price
                cash = 0
                trades += 1

            # 매도 조건 체크
            conditions_met = 0
            if df.iloc[i]["short_ma"] >= df.iloc[i]["long_ma"]:
                conditions_met += 1
            if df.iloc[i]["rsi"] >= 70:
                conditions_met += 1
            if df.iloc[i]["trade_price"] >= df.iloc[i]["bb_upper"]:
                conditions_met += 1

            if holdings > 0 and conditions_met >= 2:
                cash = holdings * current_price * (1 - self.commission_rate)
                holdings = 0
                trades += 1

            # 연말 수익률 계산
            year = df.iloc[i]["datetime"].year
            if i == len(df) - 1 or df.iloc[i + 1]["datetime"].year != year:
                total_value = cash + (holdings * current_price)

                # 첫 해는 초기 자본금 대비 수익률 계산
                if not yearly_returns:
                    yearly_return = (total_value / self.initial_capital - 1) * 100
                else:
                    # 이후 연도는 전년 대비 수익률 계산
                    prev_year = max(yearly_returns.keys())
                    yearly_return = (
                        total_value / yearly_returns[prev_year]["value"] - 1
                    ) * 100

                yearly_returns[year] = {"value": total_value, "return": yearly_return}

        # 최종 결과 계산
        final_value = cash + (holdings * df.iloc[-1]["trade_price"])
        final_return = ((final_value / self.initial_capital) - 1) * 100

        # 연도별 수익/손실 집계
        profit_years = sum(
            1 for year in yearly_returns if yearly_returns[year]["return"] > 0
        )
        loss_years = sum(
            1 for year in yearly_returns if yearly_returns[year]["return"] < 0
        )

        return {
            "short_period": n,
            "long_period": m,
            "final_value": final_value,
            "final_return": final_return,
            "trades": trades,
            "profit_years": profit_years,
            "loss_years": loss_years,
            "yearly_returns": yearly_returns,
        }


def main():
    # 설정
    simulator = InvestmentSimulator()
    results = []
    best_result = None

    # 시뮬레이션 실행
    for n in range(2, 21):
        for m in range(4, 61):
            if m < n * 2:
                continue

            result = simulator.simulate("candles_days.csv", n, m)

            # 결과 저장
            results.append(
                {
                    "short_period": n,
                    "long_period": m,
                    "final_value": result["final_value"],
                    "final_return": result["final_return"],
                    "trades": result["trades"],
                    "profit_years": result["profit_years"],
                    "loss_years": result["loss_years"],
                    **{
                        f"return_{year}": result["yearly_returns"][year]["return"]
                        for year in result["yearly_returns"]
                    },
                }
            )

            # 화면 출력
            print(
                f"n={n}, m={m}, 최종잔고={result['final_value']:,.0f}, "
                f"수익년수={result['profit_years']}, 손실년수={result['loss_years']}"
            )

            # 최고 수익률 갱신 체크
            if (
                best_result is None
                or result["final_return"] > best_result["final_return"]
            ):
                best_result = result

    # 결과 파일 저장
    df_results = pd.DataFrame(results)
    df_results.to_csv("simulation_results.csv", index=False)

    # 최고 수익률 출력
    print("\n최고 수익률 기록:")
    print(f"n={best_result['short_period']}, m={best_result['long_period']}")
    print(f"최종잔고: {best_result['final_value']:,.0f}원")
    print(f"수익률: {best_result['final_return']:.2f}%")


if __name__ == "__main__":
    main()
