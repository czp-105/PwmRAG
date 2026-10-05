from concurrent.futures import ThreadPoolExecutor


def ordered_parallel_map(function, items, workers):
    if workers == 1:
        yield from map(function, items)
        return
    with ThreadPoolExecutor(max_workers=workers) as executor:
        yield from executor.map(function, items)
