import flbenchmark.logging


class LoggerManager(object):
    _instance = None
    loggers = dict()

    def __new__(self):
        if self._instance is None:
            self._instance = super().__new__(self)
        return self._instance

    @classmethod
    def get_logger(self, rank, role=None, output_dir=None):
        if rank not in self.loggers:
            if role is None or output_dir is None:
                raise RuntimeError(
                    'The role and output_dir must be specified when creating a new logger.')

            self.loggers.setdefault(
                rank,
                flbenchmark.logging.BasicLogger(
                    id=rank,
                    agent_type=role,
                    dir=output_dir + '/log',
                )
            )
        return self.loggers[rank]
    
    @classmethod
    def reset(self):
        print('Resetting loggers...')
        del self.loggers
        self.loggers = dict()
