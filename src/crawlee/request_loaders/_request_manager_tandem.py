from __future__ import annotations

from collections import deque
from datetime import timedelta
from logging import getLogger
from typing import TYPE_CHECKING

from typing_extensions import override

from crawlee._utils.docs import docs_group
from crawlee.request_loaders import RequestManager

if TYPE_CHECKING:
    from collections.abc import Sequence

    from crawlee import Request
    from crawlee.request_loaders import RequestLoader
    from crawlee.storage_clients.models import ProcessedRequest


logger = getLogger(__name__)


@docs_group('Request loaders')
class RequestManagerTandem(RequestManager):
    """Implements a tandem behaviour for a pair of `RequestLoader` and `RequestManager`.

    In this scenario, the contents of the "loader" get transferred into the "manager", allowing processing the requests
    from both sources and also enqueueing new requests (not possible with plain `RequestManager`).

    Requests are taken from the read-only loader one at a time and migrated into the read-write manager. The source
    request is marked as handled only after the manager write succeeds. A failed migration (the manager's batch write
    raised or refused the request) keeps the request in an in-memory retry queue: it is retried on the next
    `fetch_next_request` call and remains recoverable from the read-only source if the process is interrupted before
    the retry succeeds. Successful migrations are deduplicated by the manager, so a request redelivered after a restart
    is still processed only once.
    """

    def __init__(self, request_loader: RequestLoader, request_manager: RequestManager) -> None:
        self._read_only_loader = request_loader
        self._read_write_manager = request_manager
        # Requests fetched from the read-only loader whose migration into the manager has not succeeded yet. They are
        # retried before any new request is fetched, and they stay owned by the loader until the write is confirmed.
        self._pending_migrations: deque[Request] = deque()

    @override
    async def get_handled_count(self) -> int:
        return await self._read_write_manager.get_handled_count()

    @override
    async def get_total_count(self) -> int:
        return (await self._read_only_loader.get_total_count()) + (await self._read_write_manager.get_total_count())

    @override
    async def is_empty(self) -> bool:
        # A request waiting for a migration retry is still outstanding work, even though both sources momentarily
        # look empty: the loader already handed it out and the manager write failed.
        return (
            not self._pending_migrations
            and (await self._read_only_loader.is_empty())
            and (await self._read_write_manager.is_empty())
        )

    @override
    async def is_finished(self) -> bool:
        return (
            not self._pending_migrations
            and (await self._read_only_loader.is_finished())
            and (await self._read_write_manager.is_finished())
        )

    @override
    async def add_request(self, request: str | Request, *, forefront: bool = False) -> ProcessedRequest | None:
        return await self._read_write_manager.add_request(request, forefront=forefront)

    @override
    async def add_requests(
        self,
        requests: Sequence[str | Request],
        *,
        forefront: bool = False,
        batch_size: int = 1000,
        wait_time_between_batches: timedelta = timedelta(seconds=1),
        wait_for_all_requests_to_be_added: bool = False,
        wait_for_all_requests_to_be_added_timeout: timedelta | None = None,
    ) -> None:
        return await self._read_write_manager.add_requests(
            requests,
            forefront=forefront,
            batch_size=batch_size,
            wait_time_between_batches=wait_time_between_batches,
            wait_for_all_requests_to_be_added=wait_for_all_requests_to_be_added,
            wait_for_all_requests_to_be_added_timeout=wait_for_all_requests_to_be_added_timeout,
        )

    @override
    async def fetch_next_request(self) -> Request | None:
        # Retry requests whose earlier migration failed before pulling anything new from the read-only source.
        if self._pending_migrations:
            request = self._pending_migrations[0]
            if not await self._migrate_request(request):
                return None
            self._pending_migrations.popleft()
            return await self._read_write_manager.fetch_next_request()

        if await self._read_only_loader.is_finished():
            return await self._read_write_manager.fetch_next_request()

        request = await self._read_only_loader.fetch_next_request()

        if not request:
            return await self._read_write_manager.fetch_next_request()

        if not await self._migrate_request(request):
            # The write failed: keep the request owned by the read-only loader (it is not marked handled) and remember
            # it for an immediate retry on the next call. If the process dies first, the loader re-delivers it after
            # restart instead of the request being silently dropped.
            self._pending_migrations.append(request)
            return None

        return await self._read_write_manager.fetch_next_request()

    async def _migrate_request(self, request: Request) -> bool:
        """Try to move one request into the read-write manager and release it from the loader on success.

        Returns `True` when the manager accepted the request and `False` when the write failed or was refused. On the
        `False` path the read-only loader state is never changed, so the request stays recoverable there.
        """
        try:
            processed_request = await self._read_write_manager.add_request(request, forefront=True)
        except Exception:
            logger.exception(
                'Adding request from the RequestLoader to the RequestManager failed; '
                'the request will be retried and remains recoverable from the RequestLoader.',
                extra={'url': request.url, 'unique_key': request.unique_key},
            )
            return False

        if processed_request is None:
            logger.warning(
                'The RequestManager refused to store a request from the RequestLoader; the request will be retried '
                'and remains recoverable from the RequestLoader.',
                extra={'url': request.url, 'unique_key': request.unique_key},
            )
            return False

        # The request is durably stored in the manager, so it is now safe to release it from the read-only source.
        await self._read_only_loader.mark_request_as_handled(request)
        return True

    @override
    async def reclaim_request(self, request: Request, *, forefront: bool = False) -> None:
        await self._read_write_manager.reclaim_request(request, forefront=forefront)

    @override
    async def mark_request_as_handled(self, request: Request) -> None:
        await self._read_write_manager.mark_request_as_handled(request)

    async def close(self) -> None:
        """Release resources held by the underlying read-only loader, if it exposes a `close` coroutine.

        The read-write manager owns durable storage and is released separately through `drop`.
        """
        close = getattr(self._read_only_loader, 'close', None)
        if close is not None:
            await close()

    @override
    async def drop(self) -> None:
        self._pending_migrations.clear()
        await self._read_write_manager.drop()
