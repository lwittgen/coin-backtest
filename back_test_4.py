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
# 단기 이동평균선 (n개 이동평균)과 장기 이동평균선 (m개 이동평균)을 계산해야 함.
# n은 2개부터 20개까지, m은 20개부터 60개까지 변화시키면서 투자 시뮬레이션을 진행해야 함. 단, m이 n의 2배 미만인 경우는 제외.
# 직전 단기 이평선이 장기 이평선 미만이었지만, 현재 단기 이평선이 장기 이평선 이상으로 올라갈 때 매수하고, 반대로 직전 단기 이평선이 장기 이평선 이상이었지만, 현재 단기 이평선이 장기 이평선 미만으로 내려갈 때 매도.
# 매매를 할때마다 매매 금액의 일정 비율로 수수료가 발생하는 데, 기본값은 1.0%로 지정.
# 매수할때는 가진 현금으로 (수수료를 포함해) 가능한 최대 양만큼 매수, 매도할때는 전액 매도.
# 매수 수량은 자연수가 아니라 소수도 가능.

# 출력 파일: csv 파일.
# 각 줄마다 주어진 n과 m에 대한 시뮬레이션 결과를 저장.
# 최종 잔고, 최종 수익률, 각 연도별 수익률을 저장.
# 출려되는 잔고는 보유 현금과 보유 자산의 평가 금액을 합산한 금액.
# 연도별 수익률은 전년 마지막 날의 잔고 (현금 + 자산 평가 금액) 대비 올해 마지막 날의 잔고의 수익률.
# 최종 잔고 역시 현금과 평가 금액의 합산이며, 최종 수익률은 최종 잔고 대비 초기 자본금의 수익률.

# 화면 출력:
# 주어진 n과 m에 대한 시뮬레이션이 끝날때마다 csv와 동일하게 화면에도 출력.
# 최고의 수익률을 기록한 n과 m및 최종 잔고와 수익률을 출력.

# n=6, m=41
# 최종 잔고: 604,578,628원
# 총 수익률: 60357.86%


class InvestmentSimulator:
    def __init__(self, data_file, initial_capital=1_000_000, fee_rate=0.01):
        # CSV 파일에서 필요한 컬럼만 로드
        self.df = pd.read_csv(data_file, usecols=["datetime", "trade_price"])
        self.df["datetime"] = pd.to_datetime(self.df["datetime"])
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.best_result = {"n": 0, "m": 0, "final_balance": 0, "total_return": 0}

    def calculate_moving_averages(self, n, m):
        """단기 및 장기 이동평균선 계산"""
        self.df[f"MA{n}"] = self.df["trade_price"].rolling(window=n).mean()
        self.df[f"MA{m}"] = self.df["trade_price"].rolling(window=m).mean()

    def simulate(self, n, m):
        """주어진 n, m에 대한 투자 시뮬레이션 실행"""
        self.calculate_moving_averages(n, m)

        # 초기화
        cash = self.initial_capital
        holdings = 0
        trades = []

        # 이동평균선이 계산되기 전의 데이터는 제외
        start_idx = max(n, m)

        for i in range(start_idx + 1, len(self.df)):
            current_price = self.df.iloc[i]["trade_price"]
            prev_short_ma = self.df.iloc[i - 1][f"MA{n}"]
            prev_long_ma = self.df.iloc[i - 1][f"MA{m}"]
            curr_short_ma = self.df.iloc[i][f"MA{n}"]
            curr_long_ma = self.df.iloc[i][f"MA{m}"]

            # 매수 신호
            if (
                prev_short_ma < prev_long_ma
                and curr_short_ma >= curr_long_ma
                and cash > 0
            ):
                purchase_amount = cash / (1 + self.fee_rate)
                holdings = purchase_amount / current_price
                cash = 0

            # 매도 신호
            elif (
                prev_short_ma >= prev_long_ma
                and curr_short_ma < curr_long_ma
                and holdings > 0
            ):
                sale_amount = holdings * current_price
                cash = sale_amount * (1 - self.fee_rate)
                holdings = 0

            # 거래 기록 저장
            trades.append(
                {
                    "datetime": self.df.iloc[i]["datetime"],
                    "price": current_price,
                    "cash": cash,
                    "holdings": holdings,
                    "total_value": cash + (holdings * current_price),
                }
            )

        return self.calculate_returns(trades, n, m)

    def calculate_returns(self, trades, n, m):
        """수익률 계산"""
        trades_df = pd.DataFrame(trades)
        trades_df["year"] = trades_df["datetime"].dt.year

        # 연도별 수익률 계산
        yearly_returns = {}
        years = sorted(trades_df["year"].unique())

        for i in range(1, len(years)):
            prev_year = years[i - 1]
            curr_year = years[i]

            prev_year_end = trades_df[trades_df["year"] == prev_year][
                "total_value"
            ].iloc[-1]
            curr_year_end = trades_df[trades_df["year"] == curr_year][
                "total_value"
            ].iloc[-1]

            yearly_return = (curr_year_end / prev_year_end - 1) * 100
            yearly_returns[curr_year] = yearly_return

        final_balance = trades_df["total_value"].iloc[-1]
        total_return = (final_balance / self.initial_capital - 1) * 100

        # 최고 수익률 갱신 확인
        if total_return > self.best_result["total_return"]:
            self.best_result = {
                "n": n,
                "m": m,
                "final_balance": final_balance,
                "total_return": total_return,
            }

        result = {
            "n": n,
            "m": m,
            "final_balance": final_balance,
            "total_return": total_return,
            "yearly_returns": yearly_returns,
        }

        self.print_result(result)
        return result

    def print_result(self, result):
        """결과 출력"""
        print(
            f"n={result['n']}, m={result['m']}, 최종잔고={result['final_balance']:,.0f}원, 총수익률={result['total_return']:,.2f}%"
        )
        # print(f"\nn={result['n']}, m={result['m']}")
        # print(f"최종 잔고: {result['final_balance']:,.0f}원")
        # print(f"총 수익률: {result['total_return']:.2f}%")
        # print("연도별 수익률:")
        # for year, ret in result["yearly_returns"].items():
        #     print(f"{year}년: {ret:.2f}%")
        # print("-" * 50)

    def run_all_simulations(self):
        """모든 n, m 조합에 대한 시뮬레이션 실행"""
        results = []

        for n in range(2, 21):
            for m in range(4, 61):
                if m >= 2 * n:  # m이 n의 2배 이상인 경우만 실행
                    result = self.simulate(n, m)
                    results.append(result)

        self.save_results(results)
        self.print_best_result()

    def save_results(self, results):
        """결과를 CSV 파일로 저장"""
        with open("simulation_results.csv", "w", newline="") as f:
            writer = csv.writer(f)

            # 헤더 작성
            header = ["n", "m", "final_balance", "total_return"]
            years = sorted(set().union(*[r["yearly_returns"].keys() for r in results]))
            header.extend([f"{year}_return" for year in years])
            writer.writerow(header)

            # 데이터 작성
            for r in results:
                row = [r["n"], r["m"], r["final_balance"], r["total_return"]]
                row.extend([r["yearly_returns"].get(year, "") for year in years])
                writer.writerow(row)

    def print_best_result(self):
        """최고 수익률 결과 출력"""
        print("\n=== 최고 수익률 기록 ===")
        print(f"n={self.best_result['n']}, m={self.best_result['m']}")
        print(f"최종 잔고: {self.best_result['final_balance']:,.0f}원")
        print(f"총 수익률: {self.best_result['total_return']:.2f}%")


# 실행 예시
if __name__ == "__main__":
    simulator = InvestmentSimulator("candles_days.csv")
    simulator.run_all_simulations()
