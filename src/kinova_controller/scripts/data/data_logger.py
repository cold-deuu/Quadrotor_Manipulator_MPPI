import os
import csv
import numpy as np
from datetime import datetime

class DataLogger:
    def __init__(self, save_path):
        self.save_path = save_path
        self.buffer = []
        self.header = None

        # 폴더 이름을 초기화 시에만 생성 (한 번만)
        now = datetime.now()
        time_folder_name = now.strftime("%m-%d_%H_%M")
        self.time_folder_path = os.path.join(self.save_path, time_folder_name)
        os.makedirs(self.time_folder_path, exist_ok=True)

    def _flatten_dict(self, row_dict):
        flat = {}
        for k, v in row_dict.items():
            if isinstance(v, (np.ndarray, list, tuple)) and not isinstance(v, str):
                if k.endswith("_quat") and len(v) == 4:
                    subkeys = [f"{k}_x", f"{k}_y", f"{k}_z", f"{k}_w"]
                elif k.endswith("_pos") and len(v) == 3:
                    subkeys = [f"{k}_x", f"{k}_y", f"{k}_z"]
                elif k.endswith("_rpy") and len(v) == 3:
                    subkeys = [f"{k}_roll", f"{k}_pitch", f"{k}_yaw"]
                elif k == "q" and len(v) == 7:
                    subkeys = [f"q{i+1}" for i in range(7)]
                else:
                    subkeys = [f"{k}_{i}" for i in range(len(v))]
                for key, val in zip(subkeys, v):
                    flat[key] = float(val)
            else:
                flat[k] = v
        return flat

    def append(self, row_dict):
        flat = self._flatten_dict(row_dict)
        if self.header is None:
            self.header = list(flat.keys())
        else:
            for k in flat.keys():
                if k not in self.header:
                    self.header.append(k)
        self.buffer.append(flat)

    def save(self, log_name):
        if not self.buffer:
            print("No data to save.")
            return

        save_file_path = os.path.join(self.time_folder_path, f"{log_name}.csv")
        with open(save_file_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.header)
            writer.writeheader()
            writer.writerows(self.buffer)
