"""Regression tests for `RequestManagerTandem` migration and recovery semantics.

The tandem moves requests one by one from a read-only `RequestLoader` into a read-write `RequestManager`.
These tests use deterministic in-memory storage doubles so that a transient failure of the target manager's
batch write can be reproduced without real storage, networking or timing assumptions.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

from typing_extensions import override

from crawlee import Request
from crawlee.request_loaders import RequestLoader, RequestManager, RequestManagerTandem
from crawlee.storage_clients.models import ProcessedRequest

if TYPE_CHECKING:
    from collections.abc import Sequence


class FakeReadOnlyLoader(RequestLoader):
    """Deterministic in-memory stand-in for a read-only request source (e.g. `SitemapRequestLoader`).

    Fetched requests move into `in_progress` and stay there until they are marked handled, mirroring the
    persistence contract of the real loaders: a request removed from the queue but never marked handled
    is re-delivered after a restart.
    """

    def __init__(self, urls: Sequence[str]) -> None:
        self._pending: deque[Request] = deque(Request.from_url(url) for url in urls)
        self._in_progress: dict[str, Request] = {}
        self._handled: set[str] = set()
        self.closed = False

    @property
    def in_progress_urls(self) -> set[str]:
        return set(self._in_progress)

    @property
    def handled_urls(self) -> set[str]:
        return set(self._handled)

    @property
    def pending_urls(self) -> list[str]:
        return [request.url for request in self._pending]

    @override
    async def get_handled_count(self) -> int:
        return len(self._handled)

    @override
    async def get_total_count(self) -> int:
        return len(self._pending) + len(self._in_progress) + len(self._handled)

    @override
    async def is_empty(self) -> bool:
        return not self._pending

    @override
    async def is_finished(self) -> bool:
        return not self._pending and not self._in_progress

    @override
    async def fetch_next_request(self) -> Request | None:
        if not self._pending:
            return None
        request = self._pending.popleft()
        self._in_progress[request.url] = request
        return request

    @override
    async def mark_request_as_handled(self, request: Request) -> ProcessedRequest | None:
        self._in_progress.pop(request.url, None)
        self._handled.add(request.url)
        return None

    async def close(self) -> None:
        self.closed = True

    def simulate_restart(self) -> FakeReadOnlyLoader:
        """Recreate the loader the way a process restart would: in-flight requests go back to the queue."""
        urls = [request.url for request in self._pending]
        # In-flight requests were the oldest outstanding items, so they resume ahead of the still-queued ones.
        urls = [request.url for request in self._in_progress.values()] + urls
        # Already handled requests stay handled and are not redelivered.
        return FakeReadOnlyLoader(urls)


class FakeRequestManager(RequestManager):
    """Deterministic in-memory stand-in for a read-write request manager (e.g. `RequestQueue`).

    `fail_next_adds` makes the next N `add_request` calls raise, reproducing a transient batch-write failure.
    The manager deduplicates by `unique_key`, like a real request queue.
    """

    def __init__(self, urls: Sequence[str] | None = None) -> None:
        self._pending: deque[Request] = deque()
        self._in_progress: dict[str, Request] = {}
        self._handled: set[str] = set()
        self._known_keys: set[str] = set()
        self.fail_next_adds = 0
        self.refuse_next_adds = 0
        self.add_attempts: list[str] = []
        self.add_calls = 0
        self.dropped = False

        for url in urls or ():
            request = Request.from_url(url)
            self._pending.append(request)
            self._known_keys.add(request.unique_key)

    @property
    def stored_urls(self) -> set[str]:
        return {request.url for request in self._pending} | set(self._in_progress) | self._handled

    @property
    def pending_urls(self) -> list[str]:
        return [request.url for request in self._pending]

    @override
    async def drop(self) -> None:
        self.dropped = True
        self._pending.clear()
        self._in_progress.clear()
        self._handled.clear()
        self._known_keys.clear()

    @override
    async def add_request(self, request: str | Request, *, forefront: bool = False) -> ProcessedRequest | None:
        self.add_calls += 1
        if isinstance(request, str):
            request = Request.from_url(request)

        self.add_attempts.append(request.url)
        if self.fail_next_adds > 0:
            self.fail_next_adds -= 1
            raise RuntimeError('simulated transient batch write failure')
        if self.refuse_next_adds > 0:
            self.refuse_next_adds -= 1
            return None

        already_present = request.unique_key in self._known_keys
        self._known_keys.add(request.unique_key)
        if not already_present:
            if forefront:
                self._pending.appendleft(request)
            else:
                self._pending.append(request)

        return ProcessedRequest(
            unique_key=request.unique_key,
            was_already_present=already_present,
            was_already_handled=request.unique_key in self._handled,
        )

    @override
    async def fetch_next_request(self) -> Request | None:
        if not self._pending:
            return None
        request = self._pending.popleft()
        self._in_progress[request.unique_key] = request
        return request

    @override
    async def mark_request_as_handled(self, request: Request) -> ProcessedRequest | None:
        self._in_progress.pop(request.unique_key, None)
        self._handled.add(request.unique_key)
        return None

    @override
    async def reclaim_request(self, request: Request, *, forefront: bool = False) -> ProcessedRequest | None:
        self._in_progress.pop(request.unique_key, None)
        if forefront:
            self._pending.appendleft(request)
        else:
            self._pending.append(request)
        return ProcessedRequest(
            unique_key=request.unique_key,
            was_already_present=True,
            was_already_handled=False,
        )

    @override
    async def get_handled_count(self) -> int:
        return len(self._handled)

    @override
    async def get_total_count(self) -> int:
        return len(self._known_keys)

    @override
    async def is_empty(self) -> bool:
        return not self._pending

    @override
    async def is_finished(self) -> bool:
        return not self._pending and not self._in_progress


async def _drain(tandem: RequestManagerTandem, *, max_fetches: int = 100) -> list[str]:
    processed: list[str] = []
    fetch_calls = 0
    while not await tandem.is_finished():
        request = await tandem.fetch_next_request()
        fetch_calls += 1
        assert fetch_calls <= max_fetches, 'drain did not converge, a migration may be stuck retrying'
        # A transient migration failure makes one fetch return `None`; the next one retries the same request.
        if request is None:
            continue
        processed.append(request.url)
        await tandem.mark_request_as_handled(request)
    return processed


async def test_transient_target_write_failure_does_not_lose_request() -> None:
    """Reproduces the lost-request path: a failed target write must not mark the source request handled."""
    loader = FakeReadOnlyLoader(['https://a.example/', 'https://b.example/'])
    manager = FakeRequestManager()
    manager.fail_next_adds = 1
    tandem = RequestManagerTandem(loader, manager)

    first = await tandem.fetch_next_request()

    # The write failed, so nothing was handed out and the source must still own the request.
    assert first is None
    assert loader.handled_urls == set()
    assert loader.in_progress_urls == {'https://a.example/'}
    assert manager.stored_urls == set()
    assert not await tandem.is_finished()

    # The next attempt retries the same request and succeeds. The source releases it once the write is confirmed,
    # even though the request is not handled in the manager until the crawl processes it.
    retried = await tandem.fetch_next_request()
    assert retried is not None
    assert retried.url == 'https://a.example/'
    assert loader.handled_urls == {'https://a.example/'}
    assert loader.in_progress_urls == set()
    await tandem.mark_request_as_handled(retried)

    second = await tandem.fetch_next_request()
    assert second is not None
    assert second.url == 'https://b.example/'
    await tandem.mark_request_as_handled(second)

    assert await tandem.is_finished()
    assert loader.handled_urls == {'https://a.example/', 'https://b.example/'}
    # The failed attempt and the retry both targeted the manager, but the request was stored only once.
    assert manager.add_attempts == ['https://a.example/', 'https://a.example/', 'https://b.example/']


async def test_successful_retry_processes_each_request_once() -> None:
    """After transient failures and their retries, every request is processed exactly once end to end."""
    loader = FakeReadOnlyLoader([f'https://page-{i}.example/' for i in range(5)])
    manager = FakeRequestManager()
    # The first two migration attempts fail, so the first request is retried twice before succeeding.
    manager.fail_next_adds = 2
    tandem = RequestManagerTandem(loader, manager)

    processed = await _drain(tandem)

    assert sorted(processed) == sorted(f'https://page-{i}.example/' for i in range(5))
    assert len(processed) == len(set(processed))
    assert loader.handled_urls == set(processed)


async def test_request_pending_migration_blocks_empty_and_finished() -> None:
    """A request whose migration failed still counts as outstanding work for both state checks."""
    loader = FakeReadOnlyLoader(['https://a.example/'])
    manager = FakeRequestManager()
    manager.fail_next_adds = 1
    tandem = RequestManagerTandem(loader, manager)

    assert await tandem.fetch_next_request() is None

    assert not await tandem.is_empty()
    assert not await tandem.is_finished()

    manager.fail_next_adds = 0
    request = await tandem.fetch_next_request()
    assert request is not None
    assert request.url == 'https://a.example/'
    await tandem.mark_request_as_handled(request)

    assert await tandem.is_empty()
    assert await tandem.is_finished()


async def test_target_that_refuses_without_raising_keeps_request_recoverable() -> None:
    """A manager returning `None` from `add_request` (a refused write) is treated like a raised failure."""
    loader = FakeReadOnlyLoader(['https://a.example/'])
    manager = FakeRequestManager()
    manager.refuse_next_adds = 1
    tandem = RequestManagerTandem(loader, manager)

    assert await tandem.fetch_next_request() is None
    assert loader.handled_urls == set()
    assert loader.in_progress_urls == {'https://a.example/'}
    assert not await tandem.is_finished()

    # The next attempt is accepted by the same manager and the request is processed once.
    processed = await _drain(tandem)

    assert processed == ['https://a.example/']
    assert loader.handled_urls == {'https://a.example/'}


async def test_restart_recovers_request_whose_migration_failed() -> None:
    """A simulated process restart re-delivers a request that was never marked handled in the source."""
    loader = FakeReadOnlyLoader(['https://a.example/', 'https://b.example/'])
    manager = FakeRequestManager()
    manager.fail_next_adds = 1
    tandem = RequestManagerTandem(loader, manager)

    assert await tandem.fetch_next_request() is None
    assert loader.in_progress_urls == {'https://a.example/'}

    # Simulate a process interruption: the in-memory manager lost nothing durable (the write failed),
    # and a fresh loader restores in-flight requests to its queue.
    restarted_loader = loader.simulate_restart()
    # The in-flight request resumes ahead of the requests that were still queued.
    assert restarted_loader.pending_urls == ['https://a.example/', 'https://b.example/']

    restarted_manager = FakeRequestManager()
    restarted_tandem = RequestManagerTandem(restarted_loader, restarted_manager)
    processed = await _drain(restarted_tandem)

    assert sorted(processed) == ['https://a.example/', 'https://b.example/']


async def test_restart_after_successful_migration_does_not_duplicate() -> None:
    """A request already durably stored in the manager is processed once even if the source still redelivers it."""
    # State right after a process crash: the target manager durably holds the request, while the loader's last
    # persisted snapshot still redelivers it because the handled mark had not been flushed yet.
    manager = FakeRequestManager(['https://a.example/'])
    assert manager.pending_urls == ['https://a.example/']
    restarted_loader = FakeReadOnlyLoader(['https://a.example/'])
    tandem = RequestManagerTandem(restarted_loader, manager)

    processed = await _drain(tandem)

    assert processed == ['https://a.example/']
    # The redelivery was deduplicated by the manager's unique key: one stored request, one processing attempt.
    assert manager.add_calls == 1
    assert len(manager.stored_urls) == 1


async def test_duplicate_requests_are_deduplicated() -> None:
    """Repeated URLs from the source collapse onto one manager entry and one processing attempt."""
    loader = FakeReadOnlyLoader(['https://a.example/', 'https://a.example/'])
    manager = FakeRequestManager()
    tandem = RequestManagerTandem(loader, manager)

    processed = await _drain(tandem)

    assert processed == ['https://a.example/']
    assert len(manager.stored_urls) == 1


async def test_close_propagates_to_loader() -> None:
    """Closing the tandem releases the underlying loader resources."""
    loader = FakeReadOnlyLoader(['https://a.example/'])
    manager = FakeRequestManager()
    tandem = RequestManagerTandem(loader, manager)

    await tandem.close()

    assert loader.closed


async def test_drop_delegates_to_target_manager() -> None:
    loader = FakeReadOnlyLoader(['https://a.example/'])
    manager = FakeRequestManager()
    tandem = RequestManagerTandem(loader, manager)

    await tandem.drop()

    assert manager.dropped
