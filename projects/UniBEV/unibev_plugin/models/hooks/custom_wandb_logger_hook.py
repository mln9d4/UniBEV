from mmcv.runner import HOOKS, WandbLoggerHook, master_only
import torch
import os
import os.path as osp
from mmcv.utils import scandir


@HOOKS.register_module()
class CustomWandbLoggerHook(WandbLoggerHook):
    """Custom Wandb Logger Hook with separate step counters for train/val."""

    def __init__(self,
                 init_kwargs=None,
                 interval=10,
                 ignore_last=True,
                 reset_flag=False,
                 commit=True,
                 by_epoch=True,
                 with_step=True,
                 log_artifact=False):
        super().__init__(
            init_kwargs=init_kwargs,
            interval=interval,
            ignore_last=ignore_last,
            reset_flag=reset_flag,
            commit=commit,
            by_epoch=by_epoch,
            with_step=with_step)
        self.log_artifact = log_artifact
        # self.out_suffix = out_suffix

    @master_only
    def before_run(self, runner):
        """Initialize wandb and define separate metrics for train and val."""
        super().before_run(runner)
        # Define separate step counters for training and validation
        self.wandb.define_metric("train/*", step_metric="train_step")
        self.wandb.define_metric("val/*", step_metric="val_step")

    @master_only
    def log(self, runner):
        """Log metrics with appropriate step counter based on mode."""
        tags = self.get_loggable_tags(runner)
        if tags:
            # Add appropriate step based on mode
            if runner.mode == 'train':
                tags['train_step'] = self.get_iter(runner)
            elif runner.mode == 'val':
                tags['val_step'] = self.get_epoch(runner)
            print(f"[CustomWandbLoggerHook] Logging mode={runner.mode}, step={'train_step' if runner.mode == 'train' else 'val_step'}={tags.get('train_step') or tags.get('val_step')}, tags={tags}")
            self.wandb.log(tags, commit=self.commit)


    @master_only
    def after_run(self, runner) -> None:
        if self.log_artifact:
            wandb_artifact = self.wandb.Artifact(
                name='latest_checkpoint', type='model')
            
            # Get all checkpoint files
            checkpoint_files = []
            for filename in scandir(runner.work_dir, ('.pth', '.pt'), True):
                local_filepath = osp.join(runner.work_dir, filename)
                checkpoint_files.append(local_filepath)
            
            # Upload only the last (most recent) checkpoint
            if checkpoint_files:
                last_checkpoint = max(checkpoint_files, key=os.path.getmtime)
                wandb_artifact.add_file(last_checkpoint)
            
            # Upload json files
            
            self.wandb.log_artifact(wandb_artifact)
        self.wandb.join()