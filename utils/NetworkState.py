from pathlib import Path

import numpy as np
import jittor as jt
import logging

"""
Define the state of a network during training.
Including the network itself, its optimizer and the scheduler if possible.
"""


class NetworkState:

    def __init__(self, network, cfg, name):
        self.logger = logging.getLogger(f"network_state.{name}")
        self.network = network
        self.name = name
        self.iterations = 0

        self.cfg = cfg

        self._init_optimizer(cfg.get("optimizer", {}))
        self._init_scheduler(cfg.get("scheduler", {}))

        self.learning_rate = self.scheduler.get_lr()

    def _init_network(self, cfg):
        pass

    def _init_optimizer(self, cfg):
        assert "type" in cfg
        optimizer_type = cfg["type"]

        self.clip_grad_norm = float(cfg.get("clip_grad_norm", -1.0))
        self.clip_weights = float(cfg.get("clip_weights", -1.0))

        if optimizer_type == "Adam":
            lr = cfg.get("lr", 1e-4)
            beta1 = cfg.get("beta1", 0.9)
            beta2 = cfg.get("beta2", 0.999)
            weight_decay = cfg.get("weight_decay", 1e-4)
            self.optimizer = jt.optim.Adam(self.network.parameters(), lr=lr, betas=(beta1, beta2),
                                           weight_decay=weight_decay)

        elif optimizer_type == "SGD":
            lr = cfg.get("lr", 1e-4)
            momentum = cfg.get("momentum", 0.9)
            weight_decay = cfg.get("weight_decay", 1e-4)
            self.optimizer = jt.optim.SGD(self.network.parameters(), lr=lr, momentum=momentum,
                                          weight_decay=weight_decay)

        else:
            raise NotImplementedError(f"Optimizer {optimizer_type} is not implemented")

    def _init_scheduler(self, cfg, last_epoch=0):
        scheduler_type = cfg["type"]

        if scheduler_type is None:
            self.scheduler = None

        elif scheduler_type == "StepLR":
            step_size = cfg.get("step_size", 1000000)
            gamma = cfg.get("gamma", 0.1)
            self.scheduler = jt.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma,
                                                    last_epoch=last_epoch)

        else:
            raise NotImplementedError(f"Scheduler {scheduler_type} is not implemented")  # TODO: Add more scheduler

    def save_to_dict(self, save_id=None):
        if save_id is None:
            save_id = self.iterations

        # save_path = Path(self.cfg.get("checkpoint_path")) / self.name / f"{save_id}.pth"
        return {
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "iterations": self.iterations,
        }

    def load_from_dict(self, state_dict):
        self.network.load_state_dict(state_dict["network"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.iterations = state_dict["iterations"]

    def prepare(self):
        self.optimizer.zero_grad()

    def step(self):
        # Clip grad norms first
        if self.clip_grad_norm > 0:
            self.optimizer.clip_grad_norm(self.clip_grad_norm)

        self.optimizer.step()
        self.scheduler.step()
        lr = self.scheduler.get_lr()

        if lr != self.learning_rate:
            self.logger.info(f'Learning rate set to {lr}.')
            self.learning_rate = lr

        # Clip weights
        if self.clip_weights > 0:
            for p in self.network.parameters():
                np.clip(p.data, -self.clip_weights, self.clip_weights, out=p.data)

        self.iterations += 1
