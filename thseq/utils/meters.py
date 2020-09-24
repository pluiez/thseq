import time

__all__ = ['SpeedMeter', 'ElapsedTimeMeter']


class BaseMeter(object):

    def __init__(self) -> None:
        super().__init__()
        self.start_time = None
        self._duration = 0

    def __getstate__(self):
        duration = self.duration + (time.time() - self.start_time)
        state = self.__dict__.copy()
        state['start_time'] = None
        state['_duration'] = duration
        return state

    def __setstate__(self, state):
        for k in state:
            self.__dict__[k] = state[k]

    @property
    def duration(self):
        if not self.started():
            return self._duration
        return time.time() - self.start_time + self._duration

    def start(self):
        self.start_time = time.time()

    def started(self):
        return self.start_time is not None


class SpeedMeter(BaseMeter):

    def __init__(self) -> None:
        super().__init__()
        self.count = 0
        self.start_time = None

    def start(self):
        if self.started():
            raise RuntimeError(f'A {SpeedMeter.__name__} can only be started once. '
                               f'Consider instantiate another meter.')
        super().start()

    def stop(self, n=1):
        if self.started():
            self.count += n

    @property
    def avg(self):
        if not self.started():
            return 0
        return self.count / self.duration


class ElapsedTimeMeter(BaseMeter):
    """Records elapsed time in seconds"""

    def __init__(self) -> None:
        super().__init__()

    def start(self):
        if self.started():
            raise RuntimeError(f'A {SpeedMeter.__name__} can only be started once. '
                               f'Consider instantiate another meter.')
        super().start()

    @property
    def elapse(self):
        return self.duration
