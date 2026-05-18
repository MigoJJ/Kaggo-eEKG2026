
import os
import pandas as pd
import numpy as np
import wfdb
from scipy.signal import resample

class UnifiedDataLoader:
    """
    Unified loader for PTB-XL and MIT-BIH datasets.
    Standardizes sampling rate to 100Hz and maps labels.
    """
    def __init__(self, target_fs=100):
        self.target_fs = target_fs

    def load_ptbxl(self, ptbxl_dir):
        """Loads and standardizes PTB-XL metadata."""
        csv_path = os.path.join(ptbxl_dir, 'ptbxl_database.csv')
        df = pd.read_csv(csv_path, index_col='ecg_id')
        
        # Dataset identifier
        df['dataset'] = 'PTB-XL'
        
        # Simplified mapping: Extract the primary superclass from scp_codes
        # Example scp_codes: "{'NORM': 100.0, 'SR': 0.0}"
        def get_primary_label(scp_str):
            try:
                import ast
                d = ast.literal_eval(scp_str)
                return max(d, key=d.get)
            except:
                return 'UNKNOWN'
        
        df['label_unified'] = df['scp_codes'].apply(get_primary_label)
        return df

    def load_mitbih(self, mitbih_dir):
        """Loads and standardizes MIT-BIH metadata (simplified)."""
        # This assumes the directory contains .hea and .dat files
        records = [f.replace('.hea', '') for f in os.listdir(mitbih_dir) if f.endswith('.hea')]
        data = []
        for r in records:
            data.append({
                'record_id': r,
                'dataset': 'MIT-BIH',
                'filename_lr': os.path.join(mitbih_dir, r),
                'label_unified': 'Arrhythmia' # Placeholder for actual annotation parsing
            })
        return pd.DataFrame(data)

    def get_signal(self, file_path, original_fs):
        """Reads signal and resamples to target_fs."""
        record = wfdb.rdrecord(file_path)
        signal = record.p_signal
        
        if original_fs != self.target_fs:
            num_samples = int(len(signal) * self.target_fs / original_fs)
            signal = resample(signal, num_samples)
            
        return signal

if __name__ == "__main__":
    loader = UnifiedDataLoader(100)
    print("🧪 Unified Data Loader Initialized.")
