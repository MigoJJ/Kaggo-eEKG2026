import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from .digitizer import ECGDigitizerV2

def generate_synthetic_ecg(length=1000, leads=12):
    """Generates a synthetic 12-lead ECG signal (sine waves + noise for simplicity)."""
    t = np.linspace(0, 10, length)
    signals = []
    for i in range(leads):
        # Basic periodic signal + some variation per lead
        sig = np.sin(2 * np.pi * 1.2 * t) + 0.5 * np.sin(2 * np.pi * 2.4 * t + i*0.1)
        sig += 0.1 * np.random.randn(length)
        signals.append(sig)
    return np.array(signals).T

def save_signal_as_pdf(signal, output_path):
    """Saves the 12-lead signal to a PDF in a 4x3 grid with no margins."""
    fig, axes = plt.subplots(4, 3, figsize=(15, 10))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0, wspace=0)
    # Red grid simulation
    for i, ax in enumerate(axes.flatten()):
        ax.plot(signal[:, i], color='black', linewidth=1.0)
        ax.set_xlim(0, len(signal))
        ax.set_ylim(np.min(signal), np.max(signal))
        ax.axis('off')
            
    plt.savefig(output_path, format='pdf', dpi=300)
    plt.close()

def calculate_metrics(original, reconstructed):
    """Calculates MSE and Correlation between original and reconstructed signals."""
    # Ensure same length and scale
    mse = np.mean((original - reconstructed)**2)
    # Correlation per lead
    corrs = []
    for i in range(original.shape[1]):
        c = np.corrcoef(original[:, i], reconstructed[:, i])[0, 1]
        corrs.append(c)
    return mse, np.mean(corrs)

def run_benchmark():
    print("🧪 Running Synthetic Digitization Benchmark (V2 Pipeline)...")
    original = generate_synthetic_ecg()
    pdf_path = "synthetic_benchmark.pdf"
    
    save_signal_as_pdf(original, pdf_path)
    print(f"✅ Synthetic PDF generated: {pdf_path}")
    
    digitizer = ECGDigitizerV2(dpi=300)
    reconstructed, q_score = digitizer.process(pdf_path)
    
    print(f"📊 Quality Score: {q_score}")
    print(f"DEBUG: Original lead 0 range: {np.min(original[:, 0]):.2f} to {np.max(original[:, 0]):.2f}")
    print(f"DEBUG: Reconstructed lead 0 range: {np.min(reconstructed[:, 0]):.2f} to {np.max(reconstructed[:, 0]):.2f}")
    
    # Scale reconstructed signal to match original range (basic normalization)
    for i in range(12):
        orig_std = np.std(original[:, i])
        recon_std = np.std(reconstructed[:, i])
        if recon_std > 0:
            reconstructed[:, i] = (reconstructed[:, i] / recon_std) * orig_std
            
    mse, avg_corr = calculate_metrics(original, reconstructed)
    
    print(f"📊 Benchmark Results:")
    print(f"  - Mean Squared Error: {mse:.4f}")
    print(f"  - Avg Correlation: {avg_corr:.4f}")
    
    # Save comparison plot
    plt.figure(figsize=(12, 6))
    plt.plot(original[:, 0], label='Original (Lead 0)', alpha=0.7)
    plt.plot(reconstructed[:, 0], label='Reconstructed (Lead 0)', linestyle='--')
    plt.legend()
    plt.title(f"Digitization Comparison (Corr: {avg_corr:.4f})")
    plt.savefig("benchmark_comparison.png")
    print("📈 Comparison plot saved as benchmark_comparison.png")
    
    if avg_corr > 0.8:
        print("✅ Benchmark Passed (High Correlation)")
    else:
        print("⚠️ Benchmark Warning (Low Correlation)")
    
    # Clean up
    if os.path.exists(pdf_path):
        os.remove(pdf_path)

if __name__ == "__main__":
    run_benchmark()
