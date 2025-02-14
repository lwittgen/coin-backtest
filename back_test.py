import pandas as pd
import numpy as np
from datetime import datetime
import csv


def run_ma_backtest(input_file, commission_pct, output_file):
    # 데이터 로드
    df = pd.read_csv(input_file)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")  # 시간순 정렬

    # 모든 MA 기간에 대한 이동평균선을 한 번에 계산
    ma_periods = range(2, 1001)
    ma_data = {
        f"MA_{period}": df["trade_price"].rolling(window=period).mean()
        for period in ma_periods
    }
    ma_df = pd.DataFrame(ma_data)

    # 원본 데이터와 MA 데이터를 결합
    df = pd.concat([df, ma_df], axis=1)

    results = []

    # 각 MA 기간에 대해 백테스트 실행
    for ma_period in ma_periods:
        ma_col = f"MA_{ma_period}"

        # 초기 설정
        initial_balance = 1000000  # 초기 자본금 100만원
        balance = initial_balance
        position = False  # False: 현금보유, True: 주식보유
        stock_amount = 0
        yearly_results = {}

        # 첫 ma_period+1개의 데이터는 건너뛰기 (이동평균 계산에 필요한 기간)
        for i in range(ma_period + 1, len(df)):
            current_year = df["datetime"].iloc[i].year
            prev_year = df["datetime"].iloc[i - 1].year

            # 매수/매도 시그널 확인
            prev_above_ma = df["trade_price"].iloc[i - 1] > df[ma_col].iloc[i - 1]
            curr_above_ma = df["trade_price"].iloc[i] > df[ma_col].iloc[i]

            # 매수 시그널 (MA 상향돌파)
            if not prev_above_ma and curr_above_ma and not position:
                buy_price = df["trade_price"].iloc[i]
                commission = buy_price * (commission_pct / 100)
                stock_amount = balance / (buy_price + commission)
                balance = 0
                position = True

            # 매도 시그널 (MA 하향돌파)
            elif prev_above_ma and not curr_above_ma and position:
                sell_price = df["trade_price"].iloc[i]
                balance = stock_amount * sell_price
                commission = balance * (commission_pct / 100)
                balance -= commission
                stock_amount = 0
                position = False

            # 연말 결산
            if current_year != prev_year and prev_year not in yearly_results:
                if position:
                    # 현재 보유 중인 주식의 가치 계산
                    current_value = stock_amount * df["trade_price"].iloc[i - 1]
                else:
                    current_value = balance

                prev_year_value = yearly_results.get(prev_year - 1, initial_balance)
                yearly_return = ((current_value / prev_year_value) - 1) * 100
                yearly_results[prev_year] = current_value

        # 마지막 포지션 정리
        final_year = df["datetime"].iloc[-1].year
        if position:
            final_value = stock_amount * df["trade_price"].iloc[-1]
        else:
            final_value = balance

        if final_year not in yearly_results:
            prev_year_value = yearly_results.get(final_year - 1, initial_balance)
            yearly_return = ((final_value / prev_year_value) - 1) * 100
            yearly_results[final_year] = final_value

        # 전체 수익률 계산
        total_return = ((final_value / initial_balance) - 1) * 100

        # 결과 저장
        result = {
            "MA_Period": ma_period,
            "Final_Balance": final_value,
            "Total_Return_Pct": total_return,
        }

        # 연도별 결과 추가
        for year in sorted(yearly_results.keys()):
            result[f"Balance_{year}"] = yearly_results[year]
            prev_year_value = yearly_results.get(year - 1, initial_balance)
            yearly_return = ((yearly_results[year] / prev_year_value) - 1) * 100
            result[f"Return_{year}_Pct"] = yearly_return

        results.append(result)

        # 진행상황 출력
        if ma_period % 100 == 0:
            print(f"Completed MA period: {ma_period}")

    # 결과를 DataFrame으로 변환하고 CSV로 저장
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

    # 가장 좋은 성과를 보인 MA 기간 출력
    best_result = results_df.loc[results_df["Total_Return_Pct"].idxmax()]
    print("\nBest performing MA period:")
    print(f"Period: {best_result['MA_Period']}")
    print(f"Final Balance: {best_result['Final_Balance']:,.0f}")
    print(f"Total Return: {best_result['Total_Return_Pct']:.2f}%")


# 사용 예시
if __name__ == "__main__":
    input_file = "candles_minutes_15.csv"  # 입력 파일명
    commission_pct = 1.0  # 수수료 (%)
    output_file = "backtest_minutes_15.csv"  # 결과 파일명

    run_ma_backtest(input_file, commission_pct, output_file)
