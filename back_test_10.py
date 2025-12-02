# back_test_10.py
#
# candles_days.csv 를 이용해 단순 이동평균 전략을 백테스트하고,
# 장기이평선 상승일수 필터(0~30일)를 바꿔가며 결과를
# backtest_10_days.csv 에 저장합니다.
#
# - 초기자본: 1,000,000원
# - 소수점 단위 매매 허용 (전액 매수 / 전액 매도)
# - 단기 이평: 11일선
# - 장기 이평: 38일선
# - 매매 수수료: 1% (매수/매도 각각)
# - 매수 조건:
#     종가 > 장기이평선 > 단기이평선
#     + (trend_days > 0 이면) 장기이평선이 trend_days 일 전보다 상승한 경우
# - 매도 조건:
#     종가가 장기이평선, 단기이평선 모두를 하회 (미만)
# - 출력 항목:
#     trend_days, final_equity, total_sell_count,
#     take_profit_sell_count, stop_loss_sell_count,
#     num_profitable_years, num_loss_years,
#     return_YYYY (각 연도 수익률, % 단위)
#     * 연도별 수익률은 전년 말 평가액 대비(첫 해만 첫날 평가액 대비)

import csv
from collections import deque
from datetime import datetime
from typing import List, Tuple, Dict, Any

INPUT_CSV = "candles_days.csv"
OUTPUT_CSV = "backtest_10_days.csv"

SHORT_WINDOW = 11
LONG_WINDOW = 38
INITIAL_CAPITAL = 1_000_000.0
FEE_RATE = 0.01


def read_candles(path: str) -> Tuple[List[datetime], List[float]]:
    """candles_days.csv 에서 날짜와 종가(trade_price)를 읽어온다."""
    dates: List[datetime] = []
    closes: List[float] = []

    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)

        # 헤더에서 인덱스 찾기 (순서가 바뀌어도 동작하도록)
        header_idx = {name: idx for idx, name in enumerate(header)}
        dt_idx = header_idx["datetime"]
        close_idx = header_idx["trade_price"]

        for row in reader:
            if not row:
                continue
            dt_str = row[dt_idx]
            close_str = row[close_idx]

            dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
            close = float(close_str)

            dates.append(dt)
            closes.append(close)

    return dates, closes


def compute_moving_averages(
    closes: List[float],
    short_window: int,
    long_window: int,
) -> Tuple[List[float], List[float]]:
    """단기/장기 단순이동평균(SMA)을 계산한다. 부족한 구간은 None."""
    n = len(closes)
    short_ma: List[Any] = [None] * n
    long_ma: List[Any] = [None] * n

    short_q: deque = deque()
    long_q: deque = deque()
    short_sum = 0.0
    long_sum = 0.0

    for i, price in enumerate(closes):
        # 단기
        short_q.append(price)
        short_sum += price
        if len(short_q) > short_window:
            short_sum -= short_q.popleft()
        if len(short_q) == short_window:
            short_ma[i] = short_sum / short_window

        # 장기
        long_q.append(price)
        long_sum += price
        if len(long_q) > long_window:
            long_sum -= long_q.popleft()
        if len(long_q) == long_window:
            long_ma[i] = long_sum / long_window

    return short_ma, long_ma


def run_backtest_for_trend_days(
    trend_days: int,
    dates: List[datetime],
    closes: List[float],
    short_ma: List[float],
    long_ma: List[float],
) -> Dict[str, Any]:
    """
    주어진 trend_days(장기이평 상승일수 필터) 값으로 백테스트를 수행한다.

    trend_days == 0 이면 상승추세 필터를 사용하지 않는다.
    trend_days > 0 이면, long_ma[t] > long_ma[t - trend_days] 인 경우에만 매수.
    """
    cash = INITIAL_CAPITAL
    position = 0.0  # 보유 수량
    last_buy_price = None  # 마지막 매수 가격 (익절/손절 구분용)

    total_sells = 0
    take_profit_sells = 0  # 익절 매도 횟수
    stop_loss_sells = 0    # 손절 매도 횟수

    year_first_equity: Dict[int, float] = {}
    year_last_equity: Dict[int, float] = {}

    n = len(closes)

    for i in range(n):
        price = closes[i]
        dt = dates[i]
        s_ma = short_ma[i]
        l_ma = long_ma[i]

        in_market = position > 0.0
        buy_signal = False
        sell_signal = False

        # 이동평균이 둘 다 존재해야 매매 가능
        if s_ma is not None and l_ma is not None:
            # 매수 신호: 시장 미보유 상태에서
            if not in_market:
                # 장기 이평 상승 필터
                trend_ok = True
                if trend_days > 0:
                    j = i - trend_days
                    if j < 0 or long_ma[j] is None:
                        trend_ok = False
                    else:
                        trend_ok = l_ma > long_ma[j]

                if trend_ok and (price > l_ma > s_ma):
                    buy_signal = True

            # 매도 신호: 시장 보유 상태에서
            if in_market:
                if price < l_ma and price < s_ma:
                    sell_signal = True

        # 매수: 전액 매수 (소수점 수량 허용)
        if buy_signal and cash > 0.0:
            invest_amount = cash * (1.0 - FEE_RATE)
            position = invest_amount / price
            last_buy_price = price
            cash = 0.0

        # 매도: 전량 매도
        if sell_signal and position > 0.0:
            gross_proceeds = position * price
            fee = gross_proceeds * FEE_RATE
            net_proceeds = gross_proceeds - fee
            cash += net_proceeds

            # 익절 / 손절 구분
            if last_buy_price is not None:
                if price > last_buy_price:
                    take_profit_sells += 1
                elif price < last_buy_price:
                    stop_loss_sells += 1
                # 같을 경우는 둘 다 카운트하지 않음

            total_sells += 1
            position = 0.0
            last_buy_price = None

        # 일별 종가 기준 평가금액 (EOD)
        equity = cash + position * price

        year = dt.year
        if year not in year_first_equity:
            year_first_equity[year] = equity
        year_last_equity[year] = equity

    # 최종 평가금액 (마지막 날 종가 기준)
    final_equity = cash + position * closes[-1]

    # 연도별 수익률 계산
    year_returns: Dict[int, float] = {}
    num_profitable_years = 0
    num_loss_years = 0

    prev_year_end = None
    for year in sorted(year_first_equity.keys()):
        end_eq = year_last_equity[year]
        if prev_year_end is None:
            start_eq = year_first_equity[year]  # 첫 해는 첫날 평가액 기준
        else:
            start_eq = prev_year_end  # 이후 해는 전해 마지막 평가액 기준

        if start_eq == 0:
            r = 0.0
        else:
            r = (end_eq / start_eq) - 1.0  # 예: 0.15 -> 15% 수익

        year_returns[year] = r

        if r > 0:
            num_profitable_years += 1
        elif r < 0:
            num_loss_years += 1

        prev_year_end = end_eq

    return {
        "trend_days": trend_days,
        "final_equity": final_equity,
        "total_sells": total_sells,
        "take_profit_sells": take_profit_sells,
        "stop_loss_sells": stop_loss_sells,
        "num_profitable_years": num_profitable_years,
        "num_loss_years": num_loss_years,
        "year_returns": year_returns,
    }


def main() -> None:
    # 데이터 로드
    dates, closes = read_candles(INPUT_CSV)

    # 이평선 계산
    short_ma, long_ma = compute_moving_averages(
        closes, SHORT_WINDOW, LONG_WINDOW
    )

    # 존재하는 연도 목록
    years = sorted({dt.year for dt in dates})

    # 출력 CSV 헤더 정의
    header = [
        "trend_days",
        "final_equity",
        "total_sell_count",
        "take_profit_sell_count",
        "stop_loss_sell_count",
        "num_profitable_years",
        "num_loss_years",
    ]
    # 연도별 수익률 (퍼센트)
    for y in years:
        header.append(f"return_{y}_pct")

    # 백테스트 실행 및 결과 저장
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for trend_days in range(0, 31):  # 0 ~ 30
            result = run_backtest_for_trend_days(
                trend_days, dates, closes, short_ma, long_ma
            )

            row = [
                result["trend_days"],
                f"{result['final_equity']:.2f}",
                result["total_sells"],
                result["take_profit_sells"],
                result["stop_loss_sells"],
                result["num_profitable_years"],
                result["num_loss_years"],
            ]

            # 연도별 수익률(%) 추가
            for y in years:
                r = result["year_returns"].get(y, 0.0)
                row.append(f"{r * 100:.4f}")  # 퍼센트로 저장

            writer.writerow(row)


if __name__ == "__main__":
    main()
