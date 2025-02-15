import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# python 투자 시뮬레이션 스크립트.

# 원본 데이터: csv 파일 (candles_days.csv)
# 각 줄마다 캔들 정보가 있음.
# datetime 컬럼은 캔들의 시작 시각.
# trade_price 컬럼은 캔들 종료 시점의 가격.
# csv 파일에서 필요한 컬럼만 로드해서 사용.
# 제일 과거 데이터부터 시간순으로 저장이 되어 있음.

# 목표: 원본 파일에 있는 과거 데이터 기준으로 투자 시뮬레이션을 진행하고, 그 결과를 출력.
# 단기 이동평균선 (11개 이동평균)과 장기 이동평균선 (38개 이동평균)을 계산해야 함.
# 단기 이평선이 장기 이평선 미만인데, 종가가 장기 이평선 초과면 매수.
# 종가가 장기 이평선 미만이고, 동시에 종가가 장기 이평선 미만이면 매도.
# 매매를 할때마다 매매 금액의 일정 비율로 수수료가 발생하는 데, 기본값은 1.0%로 지정.
# 매수할때는 가진 현금으로 (수수료를 포함해) 가능한 최대 양만큼 매수, 매도할때는 전액 매도.
# 매수 수량은 자연수가 아니라 소수도 가능.
# 초기 자본금은 백만원.

# 출력 파일: 수익률 그래프.
# 매일 매일의 잔고 그래프.
# 매일의 잔고는 보유한 현금과 보유한 자산의 평가액을 합친 값.
# 그래프는 로그 스케일.
# 비교를 위해 종가와 단기 이평선, 장기 이평선도 같이 표시.
# 좌측 y축은 자산 금액, 우측 y측은 가격 (종가, 단기이평선, 장기이평선)을 표시.
# 마우스 오버시 데이터를 읽을 수 있으면 더 좋을 듯. plotly를 사용해도 됨.


# 설정값
INITIAL_CAPITAL = 1_000_000  # 초기 자본금 (원)
COMMISSION_RATE = 0.01  # 수수료율 (1.0%)
SHORT_TERM_PERIOD = 11  # 단기 이동평균 기간
LONG_TERM_PERIOD = 38  # 장기 이동평균 기간


def load_data(filename):
    """CSV 파일에서 필요한 데이터만 로드"""
    df = pd.read_csv(filename, parse_dates=["datetime"])
    return df[["datetime", "trade_price"]]


def calculate_indicators(df):
    """이동평균선 계산"""
    df["short_ma"] = df["trade_price"].rolling(window=SHORT_TERM_PERIOD).mean()
    df["long_ma"] = df["trade_price"].rolling(window=LONG_TERM_PERIOD).mean()
    return df


def simulate_trading(df):
    """매매 시뮬레이션 실행"""
    portfolio = {
        "cash": INITIAL_CAPITAL,  # 현금
        "position": 0,  # 보유 수량
        "value": [],  # 일별 평가 금액
    }

    for i in range(len(df)):
        current_price = df.iloc[i]["trade_price"]

        # 첫 LONG_TERM_PERIOD 일은 이동평균이 없으므로 건너뜀
        if i < LONG_TERM_PERIOD:
            portfolio["value"].append(portfolio["cash"])
            continue

        short_ma = df.iloc[i]["short_ma"]
        long_ma = df.iloc[i]["long_ma"]

        # 현재 포트폴리오 가치 계산
        current_value = portfolio["cash"] + (portfolio["position"] * current_price)

        # 매수 신호: 단기 이평선이 장기 이평선 미만이고, 종가가 장기 이평선 초과
        if (
            portfolio["position"] == 0
            and short_ma < long_ma
            and current_price > long_ma
        ):
            available_cash = portfolio["cash"] / (1 + COMMISSION_RATE)
            buy_amount = available_cash / current_price
            portfolio["position"] = buy_amount
            portfolio["cash"] = 0

        # 매도 신호: 종가가 장기 이평선 미만이고, 단기 이평선도 장기 이평선 미만
        elif (
            portfolio["position"] > 0
            and current_price < long_ma
            and current_price < short_ma
        ):
            sell_value = portfolio["position"] * current_price
            portfolio["cash"] = sell_value * (1 - COMMISSION_RATE)
            portfolio["position"] = 0

        # 일별 평가 금액 기록
        portfolio["value"].append(current_value)

    df["portfolio_value"] = portfolio["value"]
    return df


def create_plot(df):
    """결과 시각화"""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 포트폴리오 가치 (로그 스케일)
    fig.add_trace(
        go.Scatter(
            x=df["datetime"],
            y=df["portfolio_value"],
            name="포트폴리오 가치",
            line=dict(color="blue"),
        ),
        secondary_y=False,
    )

    # 종가
    fig.add_trace(
        go.Scatter(
            x=df["datetime"],
            y=df["trade_price"],
            name="종가",
            line=dict(color="orange"),
        ),
        secondary_y=True,
    )

    # 단기 이동평균선
    # fig.add_trace(
    #     go.Scatter(
    #         x=df["datetime"],
    #         y=df["short_ma"],
    #         name=f"{SHORT_TERM_PERIOD}일 이동평균",
    #         line=dict(color="orange"),
    #     ),
    #     secondary_y=True,
    # )

    # # 장기 이동평균선
    # fig.add_trace(
    #     go.Scatter(
    #         x=df["datetime"],
    #         y=df["long_ma"],
    #         name=f"{LONG_TERM_PERIOD}일 이동평균",
    #         line=dict(color="red"),
    #     ),
    #     secondary_y=True,
    # )

    # 그래프 설정
    fig.update_layout(
        title="투자 시뮬레이션 결과",
        xaxis_title="날짜",
        yaxis_title="포트폴리오 가치 (원)",
        yaxis2_title="가격 (원)",
        hovermode="x unified",
    )

    # 왼쪽 y축 로그 스케일 설정
    fig.update_yaxes(type="log", secondary_y=False)

    return fig


def main():
    # 데이터 로드 및 처리
    df = load_data("candles_days.csv")
    df = calculate_indicators(df)
    df = simulate_trading(df)

    # 결과 시각화
    fig = create_plot(df)

    # HTML 파일로 저장
    fig.write_html("investment_simulation_result.html")

    # 최종 수익률 계산
    total_return = (df["portfolio_value"].iloc[-1] / INITIAL_CAPITAL - 1) * 100
    print(f"최종 수익률: {total_return:.2f}%")


if __name__ == "__main__":
    main()
