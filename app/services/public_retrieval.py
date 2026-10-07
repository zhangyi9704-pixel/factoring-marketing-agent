"""Process-local public retrieval reuse; owner-specific escalation stays outside."""
from collections import OrderedDict
import threading

_RETRIEVERS = OrderedDict()
_LOCK = threading.Lock()


class PublicRetriever:
    def __init__(self, settings):
        self.settings = settings
        self._inner = None
        self._lock = threading.Lock()

    def retrieve(self, *args, **kwargs):
        with self._lock:
            if self._inner is None:
                from app.rag.retriever import HybridRetriever
                self._inner = HybridRetriever(self.settings)
        # HybridRetriever rechecks revision and SQL catalog before cache reads.
        # It returns copies; briefing registers a separate pending per owner/session.
        return self._inner.retrieve(*args, **kwargs)


def get_public_retriever(settings):
    """Settings identity isolates injected configurations and refreshed credentials."""
    with _LOCK:
        key = id(settings)
        retriever = _RETRIEVERS.get(key)
        if retriever is None:
            retriever = PublicRetriever(settings)
            _RETRIEVERS[key] = retriever
            if len(_RETRIEVERS) > 4:
                _RETRIEVERS.popitem(last=False)
        _RETRIEVERS.move_to_end(key)
        return retriever
