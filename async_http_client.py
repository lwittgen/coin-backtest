import backoff
import httpx
import logging

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class APIResponse:
    status_code: int
    data: Any
    headers: Dict


class HTTPClientError(Exception):
    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response: Optional[Any] = None,
    ):
        self.message = message
        self.status_code = status_code
        self.response = response
        super().__init__(self.message)


class AsyncHTTPClient:
    def __init__(
        self, base_url: str, timeout: int = 30, default_headers: Optional[Dict] = None
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.default_headers = default_headers or {
            "Accept": "application/json",
            "Content-Type": "application/json; charset=utf-8",
        }

        self.client = httpx.AsyncClient(
            timeout=timeout, headers=self.default_headers, follow_redirects=True
        )

    @backoff.on_exception(
        backoff.expo,
        (httpx.TimeoutException, httpx.NetworkError),
        max_tries=3,
        max_time=30,
    )
    async def request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
        headers: Optional[Dict] = None,
    ) -> APIResponse:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        request_headers = {**self.default_headers, **(headers or {})}

        try:
            response = await self.client.request(
                method=method,
                url=url,
                params=params,
                json=data,
                headers=request_headers,
            )

            response.raise_for_status()

            return APIResponse(
                status_code=response.status_code,
                data=response.json() if response.content else None,
                headers=dict(response.headers),
            )

        except httpx.TimeoutException as e:
            logging.error(f"Request timeout: {str(e)}")
            raise HTTPClientError(f"Request timeout: {str(e)}")
        except httpx.HTTPStatusError as e:
            logging.error(f"HTTP error occurred: {str(e)}")
            raise HTTPClientError(
                f"HTTP error occurred: {str(e)}",
                status_code=e.response.status_code,
                response=e.response,
            )
        except Exception as e:
            logging.error(f"Unexpected error: {str(e)}")
            raise HTTPClientError(f"Unexpected error: {str(e)}")

    async def get(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
        headers: Optional[Dict] = None,
    ) -> APIResponse:
        return await self.request("GET", endpoint, params=params, headers=headers)

    async def post(
        self,
        endpoint: str,
        data: Dict,
        params: Optional[Dict] = None,
        headers: Optional[Dict] = None,
    ) -> APIResponse:
        return await self.request(
            "POST", endpoint, data=data, params=params, headers=headers
        )

    async def put(
        self,
        endpoint: str,
        data: Dict,
        params: Optional[Dict] = None,
        headers: Optional[Dict] = None,
    ) -> APIResponse:
        return await self.request(
            "PUT", endpoint, data=data, params=params, headers=headers
        )

    async def delete(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
        headers: Optional[Dict] = None,
    ) -> APIResponse:
        return await self.request("DELETE", endpoint, params=params, headers=headers)

    async def close(self):
        await self.client.aclose()
