"""Custom TensorBoardLogger that runs the process and opens the port."""

import socket
import subprocess
import time

from lightning.pytorch.loggers import TensorBoardLogger

from ib.utils.logging_module import logging


class CustomTensorBoardLogger(TensorBoardLogger):
    def __init__(self, save_dir: str, name: str = "", version: str = "") -> None:
        super().__init__(save_dir, name, version)
        self.tb_process = None
        self.port = self._find_free_port()
        self._launch_tensorboard()

    def _find_free_port(self):
        port = 6006
        while True:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(("localhost", port)) != 0:
                    return port
                port += 1

    def _launch_tensorboard(self) -> None:
        cmd = [
            "python",
            "-m",
            "tensorboard.main",
            "--logdir",
            self.save_dir,
            "--port",
            str(self.port),
        ]
        self.tb_process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        time.sleep(5)
        logging.panel("Tensorboard", f"TensorBoard started at: localhost:{self.port}")

    def finalize(self, status: str) -> None:
        super().finalize(status)
        if self.tb_process:
            logging.info("Shutting down TensorBoard...")
            self.tb_process.terminate()
