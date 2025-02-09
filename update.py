import requests
import pandas as pd
import os


class CandleCollector:
    def __init__(self, base_url, candle_type="days"):
        """
        캔들 데이터 수집기 초기화

        Args:
            base_url (str): API 기본 URL
            candle_type (str): 캔들 타입 ('days', 'minutes/1', 'minutes/3' 등)
        """
        self.base_url = base_url
        self.candle_type = candle_type
        self.max_limit = 200
        # 파일명에 사용할 수 없는 문자를 언더스코어로 대체
        safe_candle_type = candle_type.replace("/", "_").replace("\\", "_")
        self.csv_filename = f"candle_data_{safe_candle_type}.csv"

    def get_candles(self, to_datetime=None, count=200):
        """
        API를 통해 캔들 데이터 조회

        Args:
            to_datetime (str, optional): 조회 기준 시점 (None인 경우 현재 시점)
            count (int): 조회할 캔들 개수 (최대 200)

        Returns:
            list: 캔들 데이터 리스트
        """
        params = {"market": "KRW-BTC", "count": min(count, self.max_limit)}
        if to_datetime:
            params["to"] = to_datetime

        response = requests.get(self.base_url + self.candle_type, params=params)
        response.raise_for_status()
        return response.json()

    def get_last_datetime_from_csv(self):
        """
        CSV 파일에서 마지막 데이터의 시간 조회

        Returns:
            str: 마지막 데이터의 시간 (파일이 없는 경우 None)
        """
        if not os.path.exists(self.csv_filename):
            return None

        df = pd.read_csv(self.csv_filename)
        if df.empty:
            return None

        # datetime을 파싱하고 분 단위까지만 포함하여 변환
        last_datetime = pd.to_datetime(df["datetime"].iloc[-1])
        return last_datetime.strftime("%Y-%m-%d %H:%M")

    def collect_and_save(self):
        """
        캔들 데이터 수집 및 CSV 저장 (시간순 정렬)
        """
        # 기존 CSV 파일이 있는 경우 마지막 시점 확인
        last_datetime = self.get_last_datetime_from_csv()

        all_data = []
        to_datetime = None

        while True:
            # 캔들 데이터 조회
            candles = self.get_candles(to_datetime)

            # 더 이상 데이터가 없으면 종료
            if not candles:
                break

            # trade_price만 추출하여 데이터 저장
            # to_datetime이 None인 경우(현재 시점 조회)에만 첫 번째 캔들 제외
            start_idx = 1 if to_datetime is None else 0
            processed_data = [
                {
                    "datetime": candle["candle_date_time_kst"],
                    "opening_price": candle["opening_price"],
                    "high_price": candle["high_price"],
                    "low_price": candle["low_price"],
                    "trade_price": candle["trade_price"],
                    "candle_acc_trade_price": candle["candle_acc_trade_price"],
                    "candle_acc_trade_volume": candle["candle_acc_trade_volume"],
                }
                for candle in candles[start_idx:]
            ]

            # 이미 저장된 데이터는 제외
            if last_datetime:
                processed_data = [
                    data
                    for data in processed_data
                    if pd.to_datetime(data["datetime"]).strftime("%Y-%m-%d %H:%M")
                    > last_datetime
                ]

                # 모든 새로운 데이터를 수집했으면 종료
                if not processed_data:
                    break

            all_data.extend(processed_data)

            # 다음 조회를 위한 시점 설정
            to_datetime = candles[-1]["candle_date_time_kst"]

        if all_data:
            # 시간순으로 정렬 (오래된 데이터가 먼저 오도록)
            df = pd.DataFrame(all_data)
            df["datetime"] = pd.to_datetime(df["datetime"])  # datetime 형식으로 변환
            df = df.sort_values("datetime")  # 시간순 정렬

            # datetime을 지정된 형식으로 변환
            df["datetime"] = df["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")

            # 기존 파일이 있으면 데이터 추가, 없으면 새로 생성
            if os.path.exists(self.csv_filename):
                df.to_csv(self.csv_filename, mode="a", header=False, index=False)
            else:
                df.to_csv(self.csv_filename, index=False)

            print(
                f"Collected and saved {len(all_data)} new candles (oldest: {df['datetime'].iloc[0]}, newest: {df['datetime'].iloc[-1]})"
            )
        else:
            print("No new data to collect")


# 사용 예시
if __name__ == "__main__":
    BASE_URL = "https://api.bithumb.com/v1/candles/"
    CANDLE_TYPE = "days"  # days, minutes/1, minutes/3, minutes/5, minutes/10, minutes/15, minutes/30, minutes/60, minutes/240

    collector = CandleCollector(BASE_URL, CANDLE_TYPE)
    collector.collect_and_save()
