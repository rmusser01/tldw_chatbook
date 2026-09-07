"""TLDWAPIClient forwards its ssl_verify param into the underlying httpx client."""
import ssl

from tldw_chatbook.tldw_api.client import TLDWAPIClient


async def test_client_ssl_verify_false_disables_verification():
    client = TLDWAPIClient(base_url="https://example.invalid", ssl_verify=False)
    try:
        http = await client._get_client()
        assert http._transport._pool._ssl_context.verify_mode == ssl.CERT_NONE
    finally:
        await client.close()


async def test_client_ssl_verify_default_is_verification():
    client = TLDWAPIClient(base_url="https://example.invalid")
    try:
        http = await client._get_client()
        assert http._transport._pool._ssl_context.verify_mode == ssl.CERT_REQUIRED
    finally:
        await client.close()
