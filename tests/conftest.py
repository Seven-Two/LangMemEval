"""Offline by default: tests may use fake transports and local event-loop sockets."""
import os
import socket

import pytest


# Set before collecting modules; never load the developer's real .env in tests.
os.environ["PYTHON_DOTENV_DISABLED"] = "1"


@pytest.fixture(autouse=True)
def prevent_external_requests(monkeypatch):
    original = socket.socket.connect

    def connect(sock, address):
        # Windows asyncio may use loopback sockets for its event-loop wakeup.
        if isinstance(address, tuple) and address[0] in {"127.0.0.1", "::1", "localhost"}:
            return original(sock, address)
        raise AssertionError(f"Offline test attempted an external connection: {address!r}")

    monkeypatch.setattr(socket.socket, "connect", connect)
