from async_http_client import AsyncHTTPClient


class BithumbService:
    def __init__(self, http_client: AsyncHTTPClient):
        self.http_client = http_client

    async def get_minutes_candles(self, market, unit=1, count=1):
        """unit: 1, 3, 5, 10, 15, 30, 60, 240"""
        params = {"market": market, "count": count}
        return (
            await self.http_client.get(f"candles/minutes/{unit}", params=params)
        ).data

    async def get_days_candles(self, market, count=1):
        params = {"market": market, "count": count}
        return (await self.http_client.get("candles/days", params=params)).data
