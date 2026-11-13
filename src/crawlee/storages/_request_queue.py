from __future__ import annotations

import asyncio
from datetime import timedelta
from logging import getLogger
from typing import TYPE_CHECKING, TypeVar

from typing_extensions import override

from crawlee import Request, service_locator
from crawlee._utils.docs import docs_group
from crawlee.request_loaders import RequestManager
from crawlee.storage_clients.models import AddRequestsResponse

from ._base import Storage
from ._utils import validate_storage_name

if TYPE_CHECKING:
    from collections.abc import Sequence

    from crawlee import Request
    from crawlee.configuration import Configuration
    from crawlee.storage_clients import StorageClient
    from crawlee.storage_clients._base import RequestQueueClient
    from crawlee.storage_clients.models import ProcessedRequest, RequestQueueMetadata

logger = getLogger(__name__)

T = TypeVar('T')


@docs_group('Storages')
class RequestQueue(Storage, RequestManager):
    """Request queue is a storage for managing HTTP requests.

    The request queue class serves as a high-level interface for organizing and managing HTTP requests
    during web crawling. It provides methods for adding, retrieving, and manipulating requests throughout
    the crawling lifecycle, abstracting away the underlying storage implementation details.

    Request queue maintains the state of each URL to be crawled, tracking whether it has been processed,
    is currently being handled, or is waiting in the queue. Each URL in the queue is uniquely identified
    by a `unique_key` property, which prevents duplicate processing unless explicitly configured otherwise.

    The class supports both breadth-first and depth-first crawling strategies through its `forefront` parameter
    when adding requests. It also provides mechanisms for error handling and request reclamation when
    processing fails.

    You can open a request queue using the `open` class method, specifying either a name or ID to identify
    the queue. The underlying storage implementation is determined by the configured storage client.

    ### Usage

    ```python
    from crawlee.storages import RequestQueue

    # Open a request queue
    rq = await RequestQueue.open(name='my-queue')

    # Add a request
    await rq.add_request('https://example.com')

    # Process requests
    request = await rq.fetch_next_request()
    if request:
        try:
            # Process the request
            # ...
            await rq.mark_request_as_handled(request)
        except Exception:
            await rq.reclaim_request(request)
    ```
    """

    def __init__(self, client: RequestQueueClient, id: str, name: str | None) -> None:
        """Initialize a new instance.

        Preferably use the `RequestQueue.open` constructor to create a new instance.

        Args:
            client: An instance of a storage client.
            id: The unique identifier of the storage.
            name: The name of the storage, if available.
        """
        validate_storage_name(name)

        self._client = client
        self._id = id
        self._name = name

        self._add_requests_tasks = set[asyncio.Task[None]]()
        """A set of tasks for adding requests to the queue."""

        self._failed_add_requests_tasks = list[asyncio.Task[None]]()
        """Background add requests tasks that failed and whose failure no consumer has claimed yet.

        A failure is claimed (removed from this list) exactly once: by the `add_requests` caller waiting
        for the submission, by `is_finished` raising it, or by `cancel_pending_add_requests` logging it.
        """

    @property
    @override
    def id(self) -> str:
        return self._id

    @property
    @override
    def name(self) -> str | None:
        return self._name

    @override
    async def get_metadata(self) -> RequestQueueMetadata:
        return await self._client.get_metadata()

    @override
    async def get_handled_count(self) -> int:
        metadata = await self._client.get_metadata()
        return metadata.handled_request_count

    @override
    async def get_total_count(self) -> int:
        metadata = await self._client.get_metadata()
        return metadata.total_request_count

    @override
    @classmethod
    async def open(
        cls,
        *,
        id: str | None = None,
        name: str | None = None,
        alias: str | None = None,
        configuration: Configuration | None = None,
        storage_client: StorageClient | None = None,
    ) -> RequestQueue:
        configuration = service_locator.get_configuration() if configuration is None else configuration
        storage_client = service_locator.get_storage_client() if storage_client is None else storage_client

        client_opener_coro = storage_client.create_rq_client(id=id, name=name, alias=alias, configuration=configuration)
        additional_cache_key = storage_client.get_storage_client_cache_key(configuration=configuration)

        return await service_locator.storage_instance_manager.open_storage_instance(
            cls,
            id=id,
            name=name,
            alias=alias,
            client_opener_coro=client_opener_coro,
            storage_client_cache_key=additional_cache_key,
        )

    @override
    async def drop(self) -> None:
        # Cancel background additions first, so no pending task writes into the dropped storage afterwards.
        await self.cancel_pending_add_requests()

        # Remove from cache before dropping
        storage_instance_manager = service_locator.storage_instance_manager
        storage_instance_manager.remove_from_cache(self)

        await self._client.drop()

    @override
    async def purge(self) -> None:
        # Cancel background additions first, so no pending task writes into the purged storage afterwards.
        await self.cancel_pending_add_requests()

        try:
            await self._client.purge()
        except NotImplementedError:
            client_name = type(self._client).__name__
            if self._name is not None:
                logger.warning(
                    f'Storage client "{client_name}" does not support purging the request queue. '
                    f'Skipping purge for named queue "{self._name}" to avoid destroying persistent data; '
                    'the queue contents are left intact.'
                )
                return
            logger.warning(
                f'Storage client "{client_name}" does not support purging the request queue. '
                'Falling back to dropping and recreating the unnamed queue; the request queue ID may change.'
            )
            await self.drop()
            # Override `purge_on_start` so the storage client does not try to purge the freshly recreated
            # (and necessarily empty) queue and re-raise the same `NotImplementedError`.
            recreate_config = service_locator.get_configuration().model_copy(update={'purge_on_start': False})
            new_rq = await RequestQueue.open(configuration=recreate_config)
            self._client = new_rq._client  # noqa: SLF001
            self._id = new_rq._id  # noqa: SLF001

    @override
    async def add_request(
        self,
        request: str | Request,
        *,
        forefront: bool = False,
    ) -> ProcessedRequest | None:
        request = self._transform_request(request)
        # Route through `_process_batch` so a single add retries unprocessed requests just like a batched one.
        response = await self._process_batch([request], base_retry_wait=timedelta(seconds=1), forefront=forefront)

        if response.processed_requests:
            return response.processed_requests[0]

        # `_process_batch` already warns about requests left unprocessed after retries; only an empty response
        # (neither processed nor unprocessed) is unexpected here.
        if not response.unprocessed_requests:
            logger.warning(
                f'Request {request.url} was not processed by storage client '
                f'"{self._client.__class__.__name__}" (received an empty response).'
            )
        return None

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
        """Add requests to the queue in batches.

        The first batch is written before this method returns. The remaining batches are written by a background
        task, so that large submissions do not block the caller; `is_finished` reports the queue as not finished
        while any background task is still in progress.

        Failure semantics: requests that the storage client reports as unprocessed are retried (see
        `_process_batch`) and skipped with a warning once the retries are exhausted. If a batch raises an
        exception (a storage-level failure), the remaining batches are aborted and the failure is delivered
        to exactly one consumer: the caller waiting with `wait_for_all_requests_to_be_added=True`, the next
        `is_finished` call (which raises it instead of reporting the queue as finished), or
        `cancel_pending_add_requests`.

        Args:
            requests: Requests to enqueue.
            forefront: If True, add requests to the beginning of the queue.
            batch_size: The number of requests to add in one batch.
            wait_time_between_batches: Time to wait between adding batches.
            wait_for_all_requests_to_be_added: If True, wait for all requests to be added before returning.
                Storage-level failures of background batches are raised to the caller.
            wait_for_all_requests_to_be_added_timeout: Timeout for waiting for all requests to be added.
                If the wait times out, the background task is cancelled and `TimeoutError` is raised.

        Raises:
            TimeoutError: If `wait_for_all_requests_to_be_added_timeout` was reached before all requests
                were added.
        """
        transformed_requests = self._transform_requests(requests)
        wait_time_secs = wait_time_between_batches.total_seconds()

        # Wait for the first batch to be added
        first_batch = transformed_requests[:batch_size]
        if first_batch:
            await self._process_batch(
                first_batch,
                base_retry_wait=wait_time_between_batches,
                forefront=forefront,
            )

        # If everything fit into the first batch, there is nothing left for a background task.
        if len(transformed_requests) <= batch_size:
            return

        async def _process_remaining_batches() -> None:
            for i in range(batch_size, len(transformed_requests), batch_size):
                batch = transformed_requests[i : i + batch_size]
                await self._process_batch(
                    batch,
                    base_retry_wait=wait_time_between_batches,
                    forefront=forefront,
                )
                if i + batch_size < len(transformed_requests):
                    await asyncio.sleep(wait_time_secs)

        # Create and start the task to process remaining batches in the background
        remaining_batches_task = asyncio.create_task(
            _process_remaining_batches(),
            name='request_queue_process_remaining_batches_task',
        )

        self._add_requests_tasks.add(remaining_batches_task)
        remaining_batches_task.add_done_callback(self._on_add_requests_task_done)

        # Wait for the background task to finish if requested. `asyncio.wait_for` cancels the task both when
        # the timeout is reached and when the waiting coroutine itself is cancelled, so no ownerless task is
        # left running in either case.
        if wait_for_all_requests_to_be_added:
            try:
                await asyncio.wait_for(
                    remaining_batches_task,
                    timeout=(
                        wait_for_all_requests_to_be_added_timeout.total_seconds()
                        if wait_for_all_requests_to_be_added_timeout is not None
                        else None
                    ),
                )
            finally:
                # The caller observed the outcome of this submission, so claim its failure (if any) -
                # it must not be reported again via `is_finished`.
                self._claim_add_requests_task(remaining_batches_task)

    def _on_add_requests_task_done(self, task: asyncio.Task[None]) -> None:
        """Handle the completion of a background add requests task.

        A failed task is recorded so that its exception is delivered to exactly one consumer instead of
        being silently lost. Cancelled tasks are an intended outcome of shutdown and are not recorded.
        """
        self._add_requests_tasks.discard(task)

        if task.cancelled():
            return

        if task.exception() is not None:
            logger.debug('A background add requests task failed.', exc_info=task.exception())
            self._failed_add_requests_tasks.append(task)

    def _claim_add_requests_task(self, task: asyncio.Task[None]) -> None:
        """Mark the failure of the given task as consumed, so that it is not reported again."""
        if task in self._failed_add_requests_tasks:
            self._failed_add_requests_tasks.remove(task)

    @override
    async def cancel_pending_add_requests(self) -> None:
        pending_tasks = list(self._add_requests_tasks)

        for task in pending_tasks:
            task.cancel()

        if pending_tasks:
            # Wait for the cancellations to settle, so no task keeps writing after this method returns.
            # Task failures are already recorded in `_failed_add_requests_tasks` by the done callback,
            # so they are handled (and logged) below rather than re-raised here.
            await asyncio.gather(*pending_tasks, return_exceptions=True)

        # Claim and log failures that no other consumer has observed, so they are not silently lost.
        while self._failed_add_requests_tasks:
            failed_task = self._failed_add_requests_tasks.pop(0)
            logger.error(
                'A background add requests task failed; some requests were not added to the queue.',
                exc_info=failed_task.exception(),
            )

    async def fetch_next_request(self) -> Request | None:
        """Return the next request in the queue to be processed.

        Once you successfully finish processing of the request, you need to call `RequestQueue.mark_request_as_handled`
        to mark the request as handled in the queue. If there was some error in processing the request, call
        `RequestQueue.reclaim_request` instead, so that the queue will give the request to some other consumer
        in another call to the `fetch_next_request` method.

        Note that the `None` return value does not mean the queue processing finished, it means there are currently
        no pending requests. To check whether all requests in queue were finished, use `RequestQueue.is_finished`
        instead.

        Returns:
            The next request to process, or `None` if there are no more pending requests.
        """
        return await self._client.fetch_next_request()

    async def get_request(self, unique_key: str) -> Request | None:
        """Retrieve a specific request from the queue by its ID.

        Args:
            unique_key: Unique key of the request to retrieve.

        Returns:
            The request with the specified ID, or `None` if no such request exists.
        """
        return await self._client.get_request(unique_key)

    async def mark_request_as_handled(self, request: Request) -> ProcessedRequest | None:
        """Mark a request as handled after successful processing.

        This method should be called after a request has been successfully processed.
        Once marked as handled, the request will be removed from the queue and will
        not be returned in subsequent calls to `fetch_next_request` method.

        Args:
            request: The request to mark as handled.

        Returns:
            Information about the queue operation.
        """
        return await self._client.mark_request_as_handled(request)

    async def reclaim_request(
        self,
        request: Request,
        *,
        forefront: bool = False,
    ) -> ProcessedRequest | None:
        """Reclaim a failed request back to the queue for later processing.

        If a request fails during processing, this method can be used to return it to the queue.
        The request will be returned for processing again in a subsequent call
        to `RequestQueue.fetch_next_request`.

        Args:
            request: The request to return to the queue.
            forefront: If true, the request will be added to the beginning of the queue.
                Otherwise, it will be added to the end.

        Returns:
            Information about the queue operation.
        """
        return await self._client.reclaim_request(request, forefront=forefront)

    async def is_empty(self) -> bool:
        """Check if the request queue is empty.

        An empty queue means that there are no requests currently available to fetch. However, this does not
        necessarily mean that the crawling operation is finished, as there still might be requests being processed
        or tasks that could add additional requests to the queue.

        Returns:
            True if the request queue is empty, False otherwise.
        """
        return await self._client.is_empty()

    async def is_finished(self) -> bool:
        """Check if the request queue is finished.

        A finished queue means that all requests have been processed and there are no more tasks that could add
        additional requests to the queue. This is the definitive way to check if a crawling operation is complete.

        If a background batch of a previous `add_requests` call failed and no consumer has observed the failure
        yet, the stored exception is raised (once) instead of reporting the queue as finished - the failed
        submission means some requests were never written, so the queue must not be considered complete.

        Returns:
            True if the request queue is finished and no pending add operations, False otherwise.

        Raises:
            Exception: The exception a background `add_requests` task failed with, if it has not been
                observed yet.
        """
        if self._add_requests_tasks:
            logger.debug('Background add requests tasks are still in progress.')
            return False

        if self._failed_add_requests_tasks:
            # Only tasks that failed with an exception are recorded, so the first pop always yields one.
            failed_task = self._failed_add_requests_tasks.pop(0)
            if (error := failed_task.exception()) is not None:
                raise error

        if await self._client.is_finished():
            logger.debug('The request queue is finished.')
            return True

        return False

    async def _process_batch(
        self,
        batch: Sequence[Request],
        *,
        base_retry_wait: timedelta,
        attempt: int = 1,
        forefront: bool = False,
    ) -> AddRequestsResponse:
        """Process a batch of requests with automatic retry mechanism.

        Returns:
            A response aggregating all requests processed across attempts plus any still unprocessed once the
            retries are exhausted.
        """
        max_attempts = 5
        response = await self._client.add_batch_of_requests(batch, forefront=forefront)

        request_count = len(batch) - len(response.unprocessed_requests)
        if request_count:
            logger.debug(
                f'Added {request_count} requests to the queue. Processed requests: {response.processed_requests}'
            )

        if not response.unprocessed_requests:
            return response

        logger.debug(f'Following requests were not processed: {response.unprocessed_requests}.')
        if attempt > max_attempts:
            logger.warning(
                f'Following requests were not processed even after {max_attempts} attempts:\n'
                f'{response.unprocessed_requests}'
            )
            return response

        logger.debug('Retry to add requests.')
        unprocessed_requests_unique_keys = {request.unique_key for request in response.unprocessed_requests}
        retry_batch = [request for request in batch if request.unique_key in unprocessed_requests_unique_keys]
        await asyncio.sleep((base_retry_wait * attempt).total_seconds())
        retry_response = await self._process_batch(
            retry_batch,
            base_retry_wait=base_retry_wait,
            attempt=attempt + 1,
            forefront=forefront,
        )

        # Merge the retry outcome: processed requests accumulate, unprocessed is whatever the last attempt left.
        return AddRequestsResponse(
            processed_requests=[*response.processed_requests, *retry_response.processed_requests],
            unprocessed_requests=retry_response.unprocessed_requests,
        )
