"""
A logSync is a named collection of entries in structured log format.
# TODO: Add wandb or tensorboardX support!
"""

class LogSync:
    def __init__(self, name):
        self.name = name
        self.entries = []