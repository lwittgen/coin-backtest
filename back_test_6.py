import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple


# python 투자 시뮬레이션 스크립트.

# 원본 데이터: csv 파일 (candles_days.csv)
# 각 줄마다 캔들 정보가 있음.
# datetime 컬럼은 캔들의 시작 시각.
# trade_price 컬럼은 캔들 종료 시점의 가격.
# csv 파일에서 필요한 컬럼만 로드해서 사용.
# 제일 과거 데이터부터 시간순으로 저장이 되어 있음.

# 목표: 원본 파일에 있는 과거 데이터 기준으로 투자 시뮬레이션을 진행하고, 그 결과를 출력.
# 단기 이동평균선 (n개 이동평균)과 장기 이동평균선 (m개 이동평균)을 계산해야 함.
# n은 2개부터 20개까지, m은 4개부터 60개까지 변화시키면서 투자 시뮬레이션을 진행해야 함. 단, m이 n의 2배 미만인 경우는 제외.
# 단기 이평선이 장기 이평선 미만인데, 종가가 장기 이평선 초과면 매수.
# 단기 이평선이 장기 이평선 초과인데, 종가가 장기 이평선 미만이면 매도.
# 매매를 할때마다 매매 금액의 일정 비율로 수수료가 발생하는 데, 기본값은 1.0%로 지정.
# 매수할때는 가진 현금으로 (수수료를 포함해) 가능한 최대 양만큼 매수, 매도할때는 전액 매도.
# 매수 수량은 자연수가 아니라 소수도 가능.
# 초기 자본금은 백만원.

# 출력 파일: csv 파일.
# 각 줄마다 주어진 n과 m에 대한 시뮬레이션 결과를 저장.
# 최종 잔고, 최종 수익률, 매매수, 수익이 난 해의 수, 손실난 해의 수, 각 연도별 수익률을 저장.
# 출려되는 잔고는 보유 현금과 보유 자산의 평가 금액을 합산한 금액.
# 연도별 수익률은 전년 마지막 날의 잔고 (현금 + 자산 평가 금액) 대비 올해 마지막 날의 잔고의 수익률.
# 최종 잔고 역시 현금과 평가 금액의 합산이며, 최종 수익률은 최종 잔고 대비 초기 자본금의 수익률.

# 화면 출력:
# 주어진 n과 m에 대한 시뮬레이션이 끝날때마다 한줄로 짧게 요약해서 출력. (n, m, 최종 잔고, 수익이 난 해의 수, 손실이 난 해의 수)
# 최고의 수익률을 기록한 n과 m및 최종 잔고와 수익률을 출력.

# 단기이평선(n)=11
# 장기이평선(m)=38
# 최종잔고=1,196,558,857원
# 수익률=119555.89%


class InvestmentSimulator:
    def __init__(
        self,
        csv_path: str,
        initial_capital: float = 1_000_000,
        commission_rate: float = 0.01,
    ):
        # 데이터 로드 및 초기화
        self.data = pd.read_csv(csv_path, usecols=["datetime", "trade_price"])
        self.data["datetime"] = pd.to_datetime(self.data["datetime"])
        self.data["year"] = self.data["datetime"].dt.year
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate

    def calculate_moving_averages(
        self, prices: pd.Series, short_period: int, long_period: int
    ) -> Tuple[pd.Series, pd.Series]:
        """단기/장기 이동평균 계산"""
        short_ma = prices.rolling(window=short_period).mean()
        long_ma = prices.rolling(window=long_period).mean()
        return short_ma, long_ma

    def simulate(self, short_period: int, long_period: int) -> Dict:
        """주어진 기간으로 투자 시뮬레이션 실행"""
        prices = self.data["trade_price"]
        short_ma, long_ma = self.calculate_moving_averages(
            prices, short_period, long_period
        )

        cash = self.initial_capital
        holdings = 0
        trades_count = 0
        daily_balance = []

        for i in range(max(short_period, long_period), len(prices)):
            price = prices.iloc[i]

            # 매수 시그널: 단기 < 장기, 현재가 > 장기
            if short_ma.iloc[i] < long_ma.iloc[i] and price > long_ma.iloc[i]:
                if cash > 0:
                    holdings = (cash * (1 - self.commission_rate)) / price
                    cash = 0
                    trades_count += 1

            # 매도 시그널: 단기 > 장기, 현재가 < 장기
            elif short_ma.iloc[i] > long_ma.iloc[i] and price < long_ma.iloc[i]:
                if holdings > 0:
                    cash = holdings * price * (1 - self.commission_rate)
                    holdings = 0
                    trades_count += 1

            # 일별 잔고 기록 (현금 + 평가금액)
            total_balance = cash + (holdings * price)
            daily_balance.append(
                {
                    "datetime": self.data["datetime"].iloc[i],
                    "year": self.data["datetime"].iloc[i].year,
                    "balance": total_balance,
                }
            )

        # 결과 분석
        df_balance = pd.DataFrame(daily_balance)
        yearly_returns = self.calculate_yearly_returns(df_balance)

        final_balance = daily_balance[-1]["balance"]
        total_return = (final_balance / self.initial_capital - 1) * 100

        profitable_years = sum(1 for ret in yearly_returns.values() if ret > 0)
        losing_years = sum(1 for ret in yearly_returns.values() if ret < 0)

        return {
            "short_period": short_period,
            "long_period": long_period,
            "final_balance": final_balance,
            "total_return": total_return,
            "trades_count": trades_count,
            "profitable_years": profitable_years,
            "losing_years": losing_years,
            "yearly_returns": yearly_returns,
        }

    def calculate_yearly_returns(self, df_balance: pd.DataFrame) -> Dict[int, float]:
        """연도별 수익률 계산"""
        yearly_returns = {}
        years = df_balance["year"].unique()

        for year in years:
            year_data = df_balance[df_balance["year"] == year]
            if len(year_data) == 0:
                continue

            # 전년도 마지막 날의 잔고
            prev_year_data = df_balance[df_balance["year"] == (year - 1)]
            if len(prev_year_data) == 0:
                continue

            start_balance = prev_year_data.iloc[-1]["balance"]
            end_balance = year_data.iloc[-1]["balance"]
            yearly_returns[year] = (end_balance / start_balance - 1) * 100

        return yearly_returns


def run_simulation(csv_path: str, output_path: str):
    """시뮬레이션 실행 및 결과 저장"""
    simulator = InvestmentSimulator(csv_path)
    results = []
    best_result = None

    # n과 m 범위 설정
    for n in range(5, 21):  # 단기 이평선
        for m in range(10, 81):  # 장기 이평선
            # m이 n의 2배 미만인 경우 제외
            if m < n * 2:
                continue

            result = simulator.simulate(n, m)

            # 최고 수익률 갱신 확인
            if (
                best_result is None
                or result["total_return"] > best_result["total_return"]
            ):
                best_result = result

            # 결과 출력
            print(
                f"n={n}, m={m}, 최종잔고={result['final_balance']:,.0f}원, "
                f"수익년={result['profitable_years']}, 손실년={result['losing_years']}"
            )

            # 연도별 수익률을 문자열로 변환
            yearly_returns_str = ",".join(
                [f"Y{year}={ret:.2f}" for year, ret in result["yearly_returns"].items()]
            )

            # 결과 저장
            results.append(
                {
                    "short_period": n,
                    "long_period": m,
                    "final_balance": result["final_balance"],
                    "total_return": result["total_return"],
                    "trades_count": result["trades_count"],
                    "profitable_years": result["profitable_years"],
                    "losing_years": result["losing_years"],
                    "yearly_returns": yearly_returns_str,
                }
            )

    # 결과를 DataFrame으로 변환하여 CSV 저장
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_path, index=False)

    # 최고 수익률 결과 출력
    print("\n===== 최고 수익률 기록 =====")
    print(f"단기이평선(n)={best_result['short_period']}")
    print(f"장기이평선(m)={best_result['long_period']}")
    print(f"최종잔고={best_result['final_balance']:,.0f}원")
    print(f"수익률={best_result['total_return']:.2f}%")


if __name__ == "__main__":
    # CSV 파일 경로 설정
    input_csv = "candles_days.csv"  # 원본 데이터 파일
    output_csv = "simulation_results.csv"  # 결과 저장 파일

    run_simulation(input_csv, output_csv)
