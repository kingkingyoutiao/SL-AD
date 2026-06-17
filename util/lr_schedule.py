class LearningRateScheduler:
    def __init__(self, optimizer, args):
        self.optimizer = optimizer
        self.args = args
        self.loss_history = []
        self.best_loss = float('inf')
        self.no_improvement_count = 0

    def adjust_lr(self, epoch, train_loss):
        # 如果lr_flag为0，则不调整学习率
        if self.args.lr_flag == 0:
            return

        # 策略1：基于epoch调整学习率
        if self.args.lr_flag == 1:
            if epoch == 10:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = 0.0005
            elif epoch == 25:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = 0.0001
            elif epoch == 40:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = 0.00005
            elif epoch == 60:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = 0.00001

        elif self.args.lr_flag == 2:
            if train_loss < self.best_loss:
                self.best_loss = train_loss
                self.no_improvement_count = 0
            else:
                self.no_improvement_count += 1

            if self.no_improvement_count >= self.args.lr_decay_step:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] / 2
                print(f"Epoch {epoch}: Learning rate reduced to {param_group['lr']}\n")
                self.no_improvement_count = 0