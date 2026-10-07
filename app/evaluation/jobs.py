"""HTTP/SSE共享评估任务；客户端断开不取消已领取的落库工作。"""
from concurrent.futures import ThreadPoolExecutor
import threading
import copy

_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix='evaluation')
_JOBS = {}
_LOCK = threading.Lock()


class Job:
    def __init__(self):
        self.condition = threading.Condition()
        self.events = []
        self.done = False
        self.error = None

    def subscribe(self):
        index = 0
        while True:
            with self.condition:
                self.condition.wait_for(lambda: index < len(self.events) or self.done)
                pending = self.events[index:]
                index = len(self.events)
                done, error = self.done, self.error
            for event in pending:
                yield copy.deepcopy(event)
            if done:
                if error:
                    raise error
                return


def stream_job(key, produce):
    with _LOCK:
        job = _JOBS.get(key)
        if job is None:
            job = Job()
            _JOBS[key] = job
            def run():
                try:
                    for event in produce():
                        with job.condition:
                            job.events.append(event)
                            job.condition.notify_all()
                except Exception as exc:
                    job.error = exc
                finally:
                    with job.condition:
                        job.done = True
                        job.condition.notify_all()
                    with _LOCK:
                        _JOBS.pop(key, None)
            _POOL.submit(run)
    yield from job.subscribe()
