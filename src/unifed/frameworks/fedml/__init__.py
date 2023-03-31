import sys

from unifed.frameworks.fedml import protocol
from unifed.frameworks.fedml.workload_sim import *


def run_protocol():
    print('Running protocol...')
    protocol.pop.run()  # FIXME: require extra testing here
