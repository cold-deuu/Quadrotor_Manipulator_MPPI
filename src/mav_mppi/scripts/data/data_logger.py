import os
import csv
import numpy as np

class DataLogger:
    def __init__(self, save_path):
        self.save_path = save_path
        self.buffer = []
        self.header = None

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

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
        # robust header update: 새 필드 발견시 자동 추가
        if self.header is None:
            self.header = list(flat.keys())
        else:
            for k in flat.keys():
                if k not in self.header:
                    self.header.append(k)
        self.buffer.append(flat)

    def save(self):
        if not self.buffer:
            print("No data to save.")
            return
        with open(self.save_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.header)
            writer.writeheader()
            writer.writerows(self.buffer)
        print(f"[Saved] {self.save_path}")
