from __future__ import annotations

import base64
from typing import TYPE_CHECKING
from unittest.mock import patch

from crawlee import service_locator
from crawlee.request_loaders import RequestManagerTandem
from crawlee.request_loaders._sitemap_request_loader import SitemapRequestLoader
from crawlee.storages import RequestQueue
from tests._utils import (
    get_basic_results,
    get_basic_sitemap,
    poll_until_condition,
)

if TYPE_CHECKING:
    from yarl import URL

    from crawlee import Request, RequestOptions, RequestTransformAction
    from crawlee._types import JsonSerializable
    from crawlee.http_clients._base import HttpClient
    from crawlee.storage_clients.models import ProcessedRequest
    from crawlee.storages import KeyValueStore


def encode_base64(data: bytes) -> str:
    """Encode bytes to a base64 string."""
    return base64.b64encode(data).decode('utf-8')


async def test_sitemap_traversal(server_url: URL, http_client: HttpClient) -> None:
    sitemap_url = (server_url / 'sitemap.xml').with_query(
        base64=encode_base64(get_basic_sitemap(url=server_url).encode())
    )
    sitemap_loader = SitemapRequestLoader([str(sitemap_url)], http_client=http_client, enqueue_strategy='all')

    while not await sitemap_loader.is_finished():
        item = await sitemap_loader.fetch_next_request()

        if item:
            await sitemap_loader.mark_request_as_handled(item)

    assert await sitemap_loader.is_empty()
    assert await sitemap_loader.is_finished()
    assert await sitemap_loader.get_total_count() == 5
    assert await sitemap_loader.get_handled_count() == 5


async def test_is_empty_does_not_depend_on_fetch_next_request(server_url: URL, http_client: HttpClient) -> None:
    sitemap_url = (server_url / 'sitemap.xml').with_query(
        base64=encode_base64(get_basic_sitemap(url=server_url).encode())
    )
    sitemap_loader = SitemapRequestLoader([str(sitemap_url)], http_client=http_client, enqueue_strategy='all')

    items = []

    for _ in range(5):
        item = await sitemap_loader.fetch_next_request()
        assert item is not None
        assert not await sitemap_loader.is_finished()
        items.append(item)

    assert await sitemap_loader.is_empty()
    assert not await sitemap_loader.is_finished()

    for item in items:
        await sitemap_loader.mark_request_as_handled(item)

    assert await sitemap_loader.is_empty()

    assert await poll_until_condition(sitemap_loader.is_finished)


async def test_abort_sitemap_loading(server_url: URL, http_client: HttpClient) -> None:
    sitemap_url = (server_url / 'sitemap.xml').with_query(
        base64=encode_base64(get_basic_sitemap(url=server_url).encode())
    )
    sitemap_loader = SitemapRequestLoader(
        [str(sitemap_url)], max_buffer_size=2, http_client=http_client, enqueue_strategy='all'
    )

    item = await sitemap_loader.fetch_next_request()
    assert item is not None
    await sitemap_loader.mark_request_as_handled(item)

    assert not await sitemap_loader.is_empty()
    assert not await sitemap_loader.is_finished()

    await sitemap_loader.abort_loading()

    item = await sitemap_loader.fetch_next_request()
    assert item is not None
    await sitemap_loader.mark_request_as_handled(item)

    assert await sitemap_loader.is_finished()


async def test_create_persist_state_for_sitemap_loading(
    server_url: URL, http_client: HttpClient, key_value_store: KeyValueStore
) -> None:
    sitemap_url = (server_url / 'sitemap.xml').with_query(
        base64=encode_base64(get_basic_sitemap(url=server_url).encode())
    )
    persist_key = 'create_persist_state'
    sitemap_loader = SitemapRequestLoader(
        [str(sitemap_url)], http_client=http_client, persist_state_key=persist_key, enqueue_strategy='all'
    )
    assert await sitemap_loader.is_finished() is False

    await sitemap_loader.close()

    state_data = await key_value_store.get_value(persist_key)

    assert state_data is not None
    assert state_data['handledCount'] == 0


async def test_data_persistence_for_sitemap_loading(
    server_url: URL, http_client: HttpClient, key_value_store: KeyValueStore
) -> None:
    sitemap_url = (server_url / 'sitemap.xml').with_query(
        base64=encode_base64(get_basic_sitemap(url=server_url).encode())
    )
    persist_key = 'data_persist_state'
    sitemap_loader = SitemapRequestLoader(
        [str(sitemap_url)], http_client=http_client, persist_state_key=persist_key, enqueue_strategy='all'
    )

    # Give time to load.
    await poll_until_condition(sitemap_loader.is_empty, lambda is_empty: not is_empty, timeout=10)

    await sitemap_loader.close()

    state_data = await key_value_store.get_value(persist_key)

    assert state_data is not None
    assert state_data['handledCount'] == 0
    assert state_data['totalCount'] == 5
    assert len(state_data['urlQueue']) == 5


async def test_recovery_data_persistence_for_sitemap_loading(
    server_url: URL, http_client: HttpClient, key_value_store: KeyValueStore
) -> None:
    sitemap_url = (server_url / 'sitemap.xml').with_query(
        base64=encode_base64(get_basic_sitemap(url=server_url).encode())
    )
    persist_key = 'recovery_persist_state'
    sitemap_loader = SitemapRequestLoader(
        [str(sitemap_url)], http_client=http_client, persist_state_key=persist_key, enqueue_strategy='all'
    )

    item = await sitemap_loader.fetch_next_request()

    assert item is not None
    await sitemap_loader.mark_request_as_handled(item)

    await sitemap_loader.close()

    state_data = await key_value_store.get_value(persist_key)

    assert state_data is not None
    next_item_in_kvs = state_data['urlQueue'][0]

    sitemap_loader = SitemapRequestLoader(
        [str(sitemap_url)], http_client=http_client, persist_state_key=persist_key, enqueue_strategy='all'
    )

    item = await sitemap_loader.fetch_next_request()

    assert item is not None
    assert item.url == next_item_in_kvs


async def test_transform_request_function(server_url: URL, http_client: HttpClient) -> None:
    sitemap_url = (server_url / 'sitemap.xml').with_query(
        base64=encode_base64(get_basic_sitemap(url=server_url).encode())
    )

    def transform_request(request_options: RequestOptions) -> RequestOptions | RequestTransformAction:
        user_data: dict[str, JsonSerializable] = {'transformed': True}
        request_options['user_data'] = user_data
        return request_options

    sitemap_loader = SitemapRequestLoader(
        [str(sitemap_url)],
        http_client=http_client,
        transform_request_function=transform_request,
        enqueue_strategy='all',
    )

    extracted_urls = set()

    while not await sitemap_loader.is_finished():
        request = await sitemap_loader.fetch_next_request()

        if request:
            assert request.user_data.get('transformed') is True

            extracted_urls.add(request.url)

            await sitemap_loader.mark_request_as_handled(request)

    assert len(extracted_urls) == 5
    assert extracted_urls == get_basic_results(server_url)


async def test_transform_request_function_with_skip(server_url: URL, http_client: HttpClient) -> None:
    sitemap_url = (server_url / 'sitemap.xml').with_query(
        base64=encode_base64(get_basic_sitemap(url=server_url).encode())
    )

    def transform_request(_request_options: RequestOptions) -> RequestOptions | RequestTransformAction:
        return 'skip'

    sitemap_loader = SitemapRequestLoader(
        [str(sitemap_url)],
        http_client=http_client,
        transform_request_function=transform_request,
        enqueue_strategy='all',
    )

    while not await sitemap_loader.is_finished():
        request = await sitemap_loader.fetch_next_request()

        if request:
            await sitemap_loader.mark_request_as_handled(request)

    # Even though the sitemap had URLs, all were skipped, so the loader should be empty and finished with
    # 0 handled requests.
    assert await sitemap_loader.is_empty()
    assert await sitemap_loader.is_finished()
    assert await sitemap_loader.get_total_count() == 0
    assert await sitemap_loader.get_handled_count() == 0


async def test_sitemap_loader_to_tandem(
    server_url: URL,
    http_client: HttpClient,
) -> None:
    sitemap_url = (server_url / 'sitemap.xml').with_query(
        base64=encode_base64(get_basic_sitemap(url=server_url).encode())
    )

    sitemap_loader = SitemapRequestLoader([str(sitemap_url)], http_client=http_client, enqueue_strategy='all')
    request_manager = await sitemap_loader.to_tandem()

    while not await sitemap_loader.is_finished():
        request = await request_manager.fetch_next_request()

        if request:
            await request_manager.mark_request_as_handled(request)

    assert await sitemap_loader.is_empty()
    assert await sitemap_loader.is_finished()

    assert await request_manager.is_empty()
    assert await request_manager.is_finished()


async def test_sitemap_loader_to_tandem_recovers_from_transient_write_failure(
    server_url: URL,
    http_client: HttpClient,
) -> None:
    """A transient target write failure neither loses the request nor marks it handled in the loader."""
    sitemap_url = (server_url / 'sitemap.xml').with_query(
        base64=encode_base64(get_basic_sitemap(url=server_url).encode())
    )

    sitemap_loader = SitemapRequestLoader(
        [str(sitemap_url)],
        http_client=http_client,
        enqueue_strategy='all',
    )
    request_manager = await sitemap_loader.to_tandem()

    real_add_request = request_manager._read_write_manager.add_request
    flaky_calls = 0

    async def flaky_add_request(request: Request, *, forefront: bool = False) -> ProcessedRequest | None:
        # Fail the first migration attempt of every request, then let the retry through.
        nonlocal flaky_calls
        flaky_calls += 1
        if flaky_calls % 2 == 1:
            raise RuntimeError('Failed to add request')
        return await real_add_request(request, forefront=forefront)

    with patch.object(request_manager._read_write_manager, 'add_request', side_effect=flaky_add_request):
        processed: list[str] = []
        while not await request_manager.is_finished():
            request = await request_manager.fetch_next_request()
            if request:
                processed.append(request.url)
                await request_manager.mark_request_as_handled(request)

    expected = get_basic_results(server_url)
    assert sorted(processed) == sorted(expected)
    assert len(processed) == len(set(processed))

    assert await sitemap_loader.is_empty()
    assert await sitemap_loader.is_finished()
    assert await request_manager.is_empty()
    assert await request_manager.is_finished()


async def test_sitemap_loader_to_tandem_restart_after_failed_migration(
    server_url: URL,
    http_client: HttpClient,
) -> None:
    """A request whose target write failed is redelivered from the persisted loader state after a restart."""
    sitemap_url = (server_url / 'sitemap.xml').with_query(
        base64=encode_base64(get_basic_sitemap(url=server_url).encode())
    )
    persist_key = 'tandem_failed_migration_recovery'

    sitemap_loader = SitemapRequestLoader(
        [str(sitemap_url)],
        http_client=http_client,
        enqueue_strategy='all',
        persist_state_key=persist_key,
    )
    request_manager = await sitemap_loader.to_tandem()

    with patch.object(
        request_manager._read_write_manager, 'add_request', side_effect=RuntimeError('transient write failure')
    ):
        # The migration fails, so the tandem hands nothing out and the loader keeps the request in flight.
        assert await request_manager.fetch_next_request() is None
        assert not await sitemap_loader.is_finished()
        # The pending migration still counts as outstanding work, so the tandem is neither empty nor finished.
        assert not await request_manager.is_empty()
        assert not await request_manager.is_finished()

    # Simulate an abrupt process interruption: the in-memory retry buffer is gone, but the loader's persisted
    # state still owns the request because it was never marked handled.
    await sitemap_loader.close()

    restarted_loader = SitemapRequestLoader(
        [str(sitemap_url)],
        http_client=http_client,
        enqueue_strategy='all',
        persist_state_key=persist_key,
    )
    restarted_manager = await restarted_loader.to_tandem()

    processed: list[str] = []
    while not await restarted_manager.is_finished():
        request = await restarted_manager.fetch_next_request()
        if request:
            processed.append(request.url)
            await restarted_manager.mark_request_as_handled(request)

    assert sorted(processed) == sorted(get_basic_results(server_url))
    await restarted_loader.close()


async def test_sitemap_loader_to_tandem_restart_does_not_duplicate_migrated_request(
    server_url: URL,
    http_client: HttpClient,
) -> None:
    """A request already stored in the target queue before the crash is not processed again after restart."""
    sitemap_url = (server_url / 'sitemap.xml').with_query(
        base64=encode_base64(get_basic_sitemap(url=server_url).encode())
    )
    persist_key = 'tandem_successful_migration_recovery'
    queue_name = 'tandem-recovery-queue'

    sitemap_loader = SitemapRequestLoader(
        [str(sitemap_url)],
        http_client=http_client,
        enqueue_strategy='all',
        persist_state_key=persist_key,
    )
    request_queue = await RequestQueue.open(name=queue_name)
    request_manager = RequestManagerTandem(sitemap_loader, request_queue)

    request = await request_manager.fetch_next_request()
    assert request is not None
    migrated_url = request.url
    # The migration succeeded and the loader released the request, but processing (and thus marking handled)
    # never happened because the process died here.
    assert migrated_url not in sitemap_loader._state.current_value.in_progress
    await sitemap_loader.close()

    # Simulate a fresh process: drop the cached storage instances so the named queue is reopened from disk and
    # reclaims its in-progress request, exactly as a real restart does.
    service_locator.storage_instance_manager.clear_cache()

    restarted_loader = SitemapRequestLoader(
        [str(sitemap_url)],
        http_client=http_client,
        enqueue_strategy='all',
        persist_state_key=persist_key,
    )
    restarted_queue = await RequestQueue.open(name=queue_name)
    restarted_manager = RequestManagerTandem(restarted_loader, restarted_queue)

    processed: list[str] = []
    while not await restarted_manager.is_finished():
        next_request = await restarted_manager.fetch_next_request()
        if next_request:
            processed.append(next_request.url)
            await restarted_manager.mark_request_as_handled(next_request)

    expected = get_basic_results(server_url)
    assert sorted(processed) == sorted(expected)
    assert processed.count(migrated_url) == 1
    await restarted_loader.close()
    await restarted_queue.drop()


def _make_urlset(urls: list[str]) -> str:
    """Build a `<urlset>` sitemap XML containing the given URLs."""
    url_blocks = '\n'.join(f'<url><loc>{url}</loc></url>' for url in urls)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        f'{url_blocks}\n'
        '</urlset>'
    )


def _make_sitemapindex(sitemap_urls: list[str]) -> str:
    """Build a `<sitemapindex>` XML pointing at the given nested sitemap URLs."""
    sitemap_blocks = '\n'.join(f'<sitemap><loc>{url}</loc></sitemap>' for url in sitemap_urls)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        f'{sitemap_blocks}\n'
        '</sitemapindex>'
    )


async def test_sitemap_loader_filters_cross_host_urls(server_url: URL, http_client: HttpClient) -> None:
    """Default strategy `same-hostname` filters out content URLs that are not on the sitemap's host."""
    same_host_url = str(server_url / 'page')
    cross_host_url = 'http://other.test/payload'
    sitemap_content = _make_urlset([same_host_url, cross_host_url])
    sitemap_url = (server_url / 'sitemap.xml').with_query(base64=encode_base64(sitemap_content.encode()))

    loader = SitemapRequestLoader([str(sitemap_url)], http_client=http_client)

    fetched: list[str] = []
    while not await loader.is_finished():
        request = await loader.fetch_next_request()
        if request is not None:
            fetched.append(request.url)
            await loader.mark_request_as_handled(request)

    assert fetched == [same_host_url]


async def test_sitemap_loader_filters_cross_host_nested_sitemap(server_url: URL, http_client: HttpClient) -> None:
    """Nested `<sitemap><loc>` entries on a different host are dropped before fetching them."""
    child_content = _make_urlset([str(server_url / 'inner')])
    same_host_child_url = str((server_url / 'sitemap.xml').with_query(base64=encode_base64(child_content.encode())))
    cross_host_child_url = 'http://other.test/child.xml'
    index_content = _make_sitemapindex([same_host_child_url, cross_host_child_url])
    index_url = str((server_url / 'sitemap.xml').with_query(base64=encode_base64(index_content.encode())))

    loader = SitemapRequestLoader([index_url], http_client=http_client)

    fetched: list[str] = []
    while not await loader.is_finished():
        request = await loader.fetch_next_request()
        if request is not None:
            fetched.append(request.url)
            await loader.mark_request_as_handled(request)

    assert fetched == [str(server_url / 'inner')]


async def test_sitemap_loader_stamps_request_enqueue_strategy(server_url: URL, http_client: HttpClient) -> None:
    """Emitted `Request` objects carry the loader's enqueue strategy so redirects are policed downstream."""
    same_host_url = str(server_url / 'page')
    sitemap_content = _make_urlset([same_host_url])
    sitemap_url = (server_url / 'sitemap.xml').with_query(base64=encode_base64(sitemap_content.encode()))

    loader = SitemapRequestLoader([str(sitemap_url)], http_client=http_client, enqueue_strategy='same-domain')
    request = await loader.fetch_next_request()
    assert request is not None
    assert request.enqueue_strategy == 'same-domain'
    await loader.mark_request_as_handled(request)


async def test_sitemap_loader_strategy_all_disables_filtering(server_url: URL, http_client: HttpClient) -> None:
    """Passing `enqueue_strategy='all'` keeps the pre-fix permissive behavior for opt-in callers."""
    cross_host_url = 'http://other.test/payload'
    sitemap_content = _make_urlset([cross_host_url])
    sitemap_url = (server_url / 'sitemap.xml').with_query(base64=encode_base64(sitemap_content.encode()))

    loader = SitemapRequestLoader([str(sitemap_url)], http_client=http_client, enqueue_strategy='all')

    fetched: list[str] = []
    while not await loader.is_finished():
        request = await loader.fetch_next_request()
        if request is not None:
            fetched.append(request.url)
            await loader.mark_request_as_handled(request)

    assert fetched == [cross_host_url]


async def test_sitemap_loader_drops_non_http_scheme_under_strategy_all(
    server_url: URL, http_client: HttpClient
) -> None:
    """Even with `enqueue_strategy='all'`, sitemap entries with non-http(s) schemes are dropped."""
    http_url = 'http://other.test/page'
    sitemap_content = _make_urlset(
        [http_url, 'mailto:foo@bar.com', 'javascript:alert(1)', 'ftp://example.com/file.txt']
    )
    sitemap_url = (server_url / 'sitemap.xml').with_query(base64=encode_base64(sitemap_content.encode()))

    loader = SitemapRequestLoader([str(sitemap_url)], http_client=http_client, enqueue_strategy='all')

    fetched: list[str] = []
    while not await loader.is_finished():
        request = await loader.fetch_next_request()
        if request is not None:
            fetched.append(request.url)
            await loader.mark_request_as_handled(request)

    assert fetched == [http_url]
