import jittor as jt
from argparse import Namespace

from experiments.BaseExperiment import BaseExperiment
from utils.NetworkState import NetworkState


class FeedForwardExperiment(BaseExperiment):
    def __init__(self, args: Namespace):
        super(FeedForwardExperiment, self).__init__(args)

    def _init_network(self):
        self.network = None

    def _init_dataset(self):
        pass

    def _init_network_state(self):
        self.state = NetworkState(
            self.network,
            self.cfg,  # Contains optimizer and scheduler
            self.task
        )

    def _train_network(self, batch, i):
        self.state.prepare()
        log_scalar, log_output = self._run_network(batch, i)
        self.state.step()

        return log_scalar, log_output

    def _run_network(self, batch, i):
        raise NotImplementedError(f"_run_network() is not implemented in FeedForwardExperiment")

    def _load_model(self):
        load_path = self.checkpoint_path / f"{self.load}"
        self.logger.info(f"Loading model from: {load_path.resolve().absolute()}")

        self.state.load_from_dict(jt.load(load_path))

    def _save_model(self, epochs, iterations):
        save_path = self.checkpoint_path / f"{self.task}-e{epochs}-i{iterations}.pth"
        self.logger.info(f"Saving model to: {save_path.resolve().absolute()}")

        jt.save(self.state.save_to_dict(), save_path)

    @property
    def i(self):
        return self.state.iterations
