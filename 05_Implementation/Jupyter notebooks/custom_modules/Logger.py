import os

from torch.utils.tensorboard import SummaryWriter

class Logger():
    """
    A class for logging information to TensorBoard and otionally to a CSV file.

    Args:
        tb_summary (SummaryWriter): The TensorBoard summary writer object.
        log_csv (bool, optional): Flag indicating whether to log to a CSV file. Defaults to False.
    """

    def __init__(self, tb_summary: SummaryWriter, log_csv: bool = False):
        self.tb_summary = tb_summary
        self.log_csv = log_csv

    def log_config(self, config):
        """
        Logs metadata information from the config to TensorBoard and CSV file.

        Args:
            config: An object containing the configuration parameters. 
        """
        scalars = [
            "train_batch_size",
            "eval_batch_size",
            "num_epochs",
            "learning_rate",
            "lr_warmup_steps",
            "evaluate_2D_epochs",
            "evaluate_num_batches",
            "evaluate_3D_epochs",
            "evaluate_num_batches_3d",
            "train_only_connected_masks",
            "debug",
            "brightness_augmentation",
            "intermediate_timestep",
            "jump_n_sample",
            "jump_length",
            "gradient_accumulation_steps",
            "proportion_training_circular_masks",
            "use_min_snr_loss",
            "snr_gamma",
        ]
        texts = [
            "mixed_precision",
            "mode",
            "model",
            "noise_scheduler",
            "lr_scheduler",
            "add_lesion_technique",
            "dataset_train_path",
            "dataset_eval_path",
            "restrict_train_slices",
            "restrict_eval_slices",
        ]

        # log at tensorboard
        for scalar in scalars:
            if hasattr(config, scalar) and getattr(config, scalar) is not None:
                self.tb_summary.add_scalar(scalar, getattr(config, scalar), 0)
        for text in texts:
            if hasattr(config, text) and getattr(config, text) is not None:
                self.tb_summary.add_text(text, getattr(config, text), 0)
        if config.target_shape:
            self.tb_summary.add_scalar("target_shape_x", config.target_shape[0], 0)
            self.tb_summary.add_scalar("target_shape_y", config.target_shape[1], 0)

        # log to csv
        if config.log_csv:
            with open(os.path.join(config.output_dir, "metrics.csv"), "w") as f:
                for scalar in scalars:
                    if hasattr(config, scalar) and getattr(config, scalar) is not None:
                        f.write(f"{scalar}:{getattr(config, scalar)},")
                for text in texts:
                    if hasattr(config, text) and getattr(config, text) is not None:
                        f.write(f"{text}:{getattr(config, text)},")
                f.write("\n")

    def log_eval_metrics(self, global_step: int, metrics: dict, output_dir: str):
        """
        Logs the evaluation metrics to TensorBoard and optionally to a CSV file.

        Args:
            global_step (int): The global step value.
            metrics (dict): A dictionary containing the metrics to be logged. 
            output_dir (str): The output directory where the CSV file will be saved.
        """
        for key, value in metrics.items(): 
            self.tb_summary.add_scalar(key, value, global_step) 

        if self.log_csv:
            with open(os.path.join(output_dir, "metrics.csv"), "a") as f:
                for key, value in metrics.items():
                    f.write(f"{key}:{value},")
                f.write(f"global_step:{global_step}")
                f.write("\n")

    def log_train_metrics(self, global_step: int, logs: dict):
        """
        Logs the evaluation metrics to TensorBoard and optionally to a CSV file.

        Args:
            global_step (int): The global step value.
            logs (dict): A dictionary containing the metrics to be logged.
        """ 
        for tag, value in logs:
            if value:
                self.tb_summary.add_scalar(tag, value, global_step)



