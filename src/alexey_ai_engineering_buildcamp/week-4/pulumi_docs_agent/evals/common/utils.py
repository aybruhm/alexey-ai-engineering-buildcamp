import asyncio

from tqdm.auto import tqdm


async def map_progress(seq, func, max_concurrency=5):
    semaphore = asyncio.Semaphore(max_concurrency)

    async def run_with_semaphore(item):
        async with semaphore:
            return await func(item)

    coros = [run_with_semaphore(el) for el in seq]
    results = []
    for coro in tqdm(asyncio.as_completed(coros), total=len(coros)):
        results.append(await coro)
    return results


def fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s" if m else f"{s}s"
