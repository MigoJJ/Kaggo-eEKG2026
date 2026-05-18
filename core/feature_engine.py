
import neurokit2 as nk
import numpy as np
import pandas as pd

class ClinicalFeatureEngine:
    """
    Engine to extract medical features (HRV, EKG intervals) from raw signals using NeuroKit2.
    """
    def __init__(self, sampling_rate=100):
        self.sampling_rate = sampling_rate

    def extract_all(self, signal, lead_idx=1):
        """
        Extracts both HRV and EKG interval features from a single lead.
        Default is Lead II (idx 1).
        """
        try:
            # 1. Signal Cleaning
            cleaned = nk.ecg_clean(signal, sampling_rate=self.sampling_rate)
            
            # 2. Peak Detection
            peaks, info = nk.ecg_peaks(cleaned, sampling_rate=self.sampling_rate)
            
            # 3. HRV Analysis (Time Domain)
            hrv_time = nk.hrv_time(peaks, sampling_rate=self.sampling_rate)
            
            # 4. EKG Delineation (Extract P, Q, R, S, T waves)
            # Note: Delineation is computationally expensive, we use a simplified version for training speed
            delineation, _ = nk.ecg_delineate(cleaned, peaks, sampling_rate=self.sampling_rate, method="peak")
            
            # 5. Combine core features into a flat dictionary
            features = {
                "HRV_MeanNN": hrv_time["HRV_MeanNN"].values[0],
                "HRV_SDNN": hrv_time["HRV_SDNN"].values[0],
                "HRV_RMSSD": hrv_time["HRV_RMSSD"].values[0],
                "HRV_pNN50": hrv_time["HRV_pNN50"].values[0],
                "HRV_LFHF": 0.0, # Placeholder for frequency domain if needed
            }
            
            # Add basic interval averages if delineation succeeded
            # PR, QRS, QT intervals
            features["QRS_Duration"] = np.nanmean(delineation.get("ECG_S_Peaks", [np.nan])) - \
                                       np.nanmean(delineation.get("ECG_Q_Peaks", [np.nan]))
            
            return features
        except Exception as e:
            # Return NaNs if extraction fails
            return {k: np.nan for k in ["HRV_MeanNN", "HRV_SDNN", "HRV_RMSSD", "HRV_pNN50", "QRS_Duration"]}

    def process_batch(self, signals, lead_idx=1):
        """Processes a batch of signals and returns a DataFrame of features."""
        results = []
        for i in range(signals.shape[0]):
            feat = self.extract_all(signals[i, lead_idx, :])
            results.append(feat)
        return pd.DataFrame(results)

if __name__ == "__main__":
    # Quick Test
    engine = ClinicalFeatureEngine(100)
    dummy_signal = np.sin(np.linspace(0, 10, 1000)) # Simple sine wave as proxy
    print("🧪 Testing Feature Engine...")
    print(engine.extract_all(dummy_signal))
