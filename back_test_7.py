import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple


# 6과 동일한데, 매도조건만 변경

# 단기이평선(n)=11
# 장기이평선(m)=38
# 최종잔고=1,387,398,313원
# 수익률=138639.83%


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

            # 매도 시그널: 단기 > 현재가, 현재가 < 장기
            elif short_ma.iloc[i] > price and price < long_ma.iloc[i]:
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
