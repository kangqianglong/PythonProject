# 生成一个异步HTTP请求函数，使用aiohttp库，支持重试3次，超时5秒
import asyncio
import logging
from typing import Optional, Dict, Any

import aiohttp

logger = logging.getLogger(__name__)

async def async_http_request(
    url: str,
    method: str = 'GET',
    data: Optional[Any] = None,
    headers: Optional[Dict[str, str]] = None,
    retries: int = 3,
    timeout: float = 5.0
) -> str:
    """Async HTTP request with retry mechanism"""

    valid_methods = {'GET', 'POST', 'PUT', 'DELETE', 'PATCH'}
    if method.upper() not in valid_methods:
        raise ValueError(f"Invalid method {method}. Must be one of {valid_methods}")

    headers = headers or {}
    timeout_obj = aiohttp.ClientTimeout(total=timeout)

    async with aiohttp.ClientSession(headers=headers) as session:
        for attempt in range(1, retries + 1):
            try:
                async with session.request(
                    method=method,
                    url=url,
                    data=data,
                    timeout=timeout_obj
                ) as response:
                    response.raise_for_status()
                    return await response.text()

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.warning(f"Attempt {attempt}/{retries} failed: {str(e)}")
                if attempt == retries:
                    raise RuntimeError(f"Request failed after {retries} retries") from e

                backoff = min(2 ** attempt, 10)  # 指数退避上限10秒
                await asyncio.sleep(backoff)

    raise RuntimeError("Unexpected code path reached")  # 防御性异常
