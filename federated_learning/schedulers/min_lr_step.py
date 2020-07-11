class MinCapableStepLR:

    def __init__(self, logger, optimizer, step_size, gamma, min_lr):
        """
        :param logger: logger
        :type logger: loguru.logger
        :param optimizer:
        :type optimizer: torch.optim
        :param step_size: # of epochs between LR updates
        :type step_size: int
        :param gamma: multiplication factor for LR update
        :type gamma: float
        :param min_lr: minimum learning rate
        :type min_lr: float
        """
        self.logger = logger

        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.min_lr = min_lr

        self.epoch_idx = 0

    def step(self):
        """
        Adjust the learning rate as necessary.
        """
        self.increment_epoch_index()

        if self.is_time_to_update_lr():
            self.logger.debug("Updating LR for optimizer")

            self.update_lr()

    def is_time_to_update_lr(self):
        return self.epoch_idx % self.step_size == 0

    def update_lr(self):
        if self.optimizer.param_groups[0]['lr'] * self.gamma >= self.min_lr:
            self.optimizer.param_groups[0]['lr'] *= self.gamma
        else:
            self.logger.warning("Updating LR would place it below min LR. Skipping LR update.")

        self.logger.debug("New LR: {}".format(self.optimizer.param_groups[0]['lr']))

    def increment_epoch_index(self):
        self.epoch_idx += 1
