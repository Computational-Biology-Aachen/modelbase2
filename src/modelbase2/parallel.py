from __future__ import annotations

import multiprocessing
import pickle
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pebble
from tqdm import tqdm

from modelbase2.types import K, Tin, Tout

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Hashable


def _pickle_name(k: Hashable) -> str:
    return f"{k}.p"


def _pickle_load(file: Path) -> Any:
    with file.open("rb") as fp:
        return pickle.load(fp)  # nosec


def _pickle_save(file: Path, data: Any) -> None:
    with file.open("wb") as fp:
        pickle.dump(data, fp)


@dataclass
class Cache:
    tmp_dir: Path = Path(".cache")
    name_fn: Callable[[Any], str] = _pickle_name
    load_fn: Callable[[Path], Any] = _pickle_load
    save_fn: Callable[[Path, Any], None] = _pickle_save


def _load_or_run(
    inp: tuple[K, Tin],
    fn: Callable[[Tin], Tout],
    cache: Cache | None,
) -> tuple[K, Tout]:
    k, v = inp
    if cache is None:
        res = fn(v)
    else:
        file = cache.tmp_dir / cache.name_fn(k)
        if file.exists():
            return k, cast(Tout, cache.load_fn(file))
        res = fn(v)
        cache.save_fn(file, res)
    return k, res


def parallelise(
    fn: Callable[[Tin], Tout],
    inputs: Collection[tuple[K, Tin]],
    *,
    cache: Cache | None,
    parallel: bool = True,
    max_workers: int | None = None,
    timeout: float | None = None,
    disable_tqdm: bool = False,
    tqdm_desc: str | None = None,
) -> dict[Tin, Tout]:
    if cache is not None:
        cache.tmp_dir.mkdir(parents=True, exist_ok=True)

    if sys.platform in ["win32", "cygwin"]:
        parallel = False

    worker: Callable[[K, Tin], tuple[K, Tout]] = partial(
        _load_or_run,
        fn=fn,
        cache=cache,
    )  # type: ignore

    results: dict[Tin, Tout]
    if parallel:
        results = {}
        max_workers = (
            multiprocessing.cpu_count() if max_workers is None else max_workers
        )

        with (
            tqdm(
                total=len(inputs),
                disable=disable_tqdm,
                desc=tqdm_desc,
            ) as pbar,
            pebble.ProcessPool(max_workers=max_workers) as pool,
        ):
            future = pool.map(worker, inputs, timeout=timeout)
            it = future.result()
            while True:
                try:
                    key, value = next(it)
                    pbar.update(1)
                    results[key] = value
                except StopIteration:
                    break
                except TimeoutError:
                    pbar.update(1)
    else:
        results = dict(
            tqdm(
                map(worker, inputs),  # type: ignore
                total=len(inputs),
                disable=disable_tqdm,
                desc=tqdm_desc,
            )  # type: ignore
        )  # type: ignore
    return results
