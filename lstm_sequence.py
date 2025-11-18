# lstm_sequence.py
import pandas as pd
import numpy as np
from pathlib import Path

class LSTMPredictionSequence:
    """
    簡單版：
    - 啟動時讀取 lstm_results_full.csv
    - 每次呼叫 next() 回傳下一個 Predicted(kWh)
    - 跑到底就循環重頭開始
    """

    def __init__(self,
                 csv_path: str | None = None,
                 column: str = "Predicted (kWh)"):
        base_dir = Path(__file__).parent

        # 預設路徑：專案底下的 artifacts_lstm
        if csv_path is None:
            csv_path = base_dir / "artifacts_lstm" / "lstm_results_full.csv"

        self.df = pd.read_csv(csv_path)

        if column not in self.df.columns:
            raise ValueError(
                f"Column '{column}' not found in {csv_path}. "
                f"Available columns: {list(self.df.columns)}"
            )

        self.values = self.df[column].to_numpy(dtype=np.float32)
        self.n = len(self.values)
        self.idx = 0

        if self.n == 0:
            print("[LSTMPredictionSequence] Warning: no data in CSV!")

    def reset(self):
        """每個 episode 開始時重置序列。"""
        self.idx = 0

    def next(self) -> float:
        """取出下一個預測值，若到底就循環。"""
        if self.n == 0:
            return 0.0

        val = self.values[self.idx]
        self.idx = (self.idx + 1) % self.n  # 跑到底就從頭再來
        return float(val)
