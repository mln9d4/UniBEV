from mmcv.runner import HOOKS, Hook
import torch


@HOOKS.register_module()
class ValLossHook(Hook):
    """Computes validation losses after each training epoch without evaluation metrics.
    
    Args:
        interval (int): Validation interval (in epochs). Default: 1.
    """

    def __init__(self, interval=1):
        self.interval = interval

    def after_train_epoch(self, runner):
        """Called after each training epoch to compute validation losses."""
        if not self.every_n_epochs(runner, self.interval):
            return

        model = runner.model
        was_training = model.training
        model.eval()

        loss_sum = {}
        num_batches = 0

        # Try to get validation dataloader
        val_dataloader = None
        if hasattr(runner, 'val_dataloader'):
            print("validation dataloader found in runner.val_dataloader")
            val_dataloader = runner.val_dataloader
        elif hasattr(runner, 'data_loader'):
            print("Normal dataloader found in runner.data_loader")
            val_dataloader = runner.data_loader
        
        if val_dataloader is None:
            print("ValLossHook: no validation dataloader found; skipping.")
            return

        with torch.no_grad():
            for data_batch in val_dataloader:
                # Call forward to compute losses
                losses = model(**data_batch, return_loss=True)
                
                # Parse losses - returns (loss_total, log_vars_dict)
                _, log_vars = runner.model.module._parse_losses(losses)
                
                # Accumulate losses from log_vars
                for k, v in log_vars.items():
                    if k not in loss_sum:
                        loss_sum[k] = 0.0
                    loss_sum[k] += v
                num_batches += 1

        if num_batches == 0:
            print("ValLossHook: validation dataloader is empty; skipping.")
            return

        # Average validation losses and store in runner
        runner.val_loss_dict = {}
        for loss_name, loss_value in loss_sum.items():
            avg_loss = loss_value / num_batches
            runner.val_loss_dict[f'val/{loss_name}'] = avg_loss
        print(f"ValLossHook: computed validation losses: {runner.val_loss_dict}")

        model.train()
        
