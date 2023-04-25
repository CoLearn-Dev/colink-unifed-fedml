import flbenchmark.logging


class LoggerManager(object):
    _instance = None
    loggers = dict()

    def __new__(self):
        if self._instance is None:
            self._instance = super().__new__(self)
        return self._instance

    @classmethod
    def get_logger(self, rank, role):
        if rank not in self.loggers:
            self.loggers.setdefault(
                rank,
                flbenchmark.logging.BasicLogger(
                    id=rank,
                    agent_type=role,
                )
            )
        return self.loggers[rank]


class ClientLogger(object):
    _instance = None
    logger = None

    def __new__(self, args=None):
        if self._instance is None:
            self._instance = super().__new__(self)
            self.logger = \
                flbenchmark.logging.BasicLogger(
                    id=args.rank,
                    agent_type='client',
                )
        return self._instance


class ServerLogger(object):
    _instance = None
    logger = None

    def __new__(self):
        if self._instance is None:
            self._instance = super().__new__(self)
            self.logger = \
                flbenchmark.logging.BasicLogger(
                    id=0,
                    agent_type='aggregator',
                )
        return self._instance
