"""Test-suite wide setup.

The event loop policy has to be selected before pytest-asyncio builds its first
loop, so this happens at import time of conftest -- which is the one place
where an import side effect is the intended mechanism.
"""

from __future__ import annotations

from smartdialer.core.runtime import configure_event_loop

configure_event_loop()
