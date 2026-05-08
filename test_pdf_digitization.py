import matplotlib.pyplot as plt
from ecg_digitizer_proto import process_ecg_document
import os

def test_digitization():
    pdf_path = "/home/ittia/git/Kaggo-eEKG2026/data/EKGPDFdata/ekg1.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_path} not found.")
        return

    print(f"Testing digitization for: {pdf_path}")
    
    try:
        # 1. 신호 추출
        signals = process_ecg_document(pdf_path)
        print(f"Successfully extracted {len(signals)} pages.")

        # 2. 결과 시각화 및 저장
        plt.figure(figsize=(15, 5))
        for i, sig in enumerate(signals):
            # 첫 페이지만 시각화 (예시)
            if i == 0:
                plt.plot(sig[:2000], linewidth=0.5, color='blue') # 처음 2000개 샘플만 출력
                plt.title(f"Digitized ECG Signal - Page {i+1} (Truncated)")
                plt.xlabel("Sample Index")
                plt.ylabel("Reconstructed Amplitude")
                plt.grid(True, alpha=0.3)
                
        output_plot = "digitized_output.png"
        plt.savefig(output_plot)
        print(f"Result plot saved as: {output_plot}")

    except Exception as e:
        print(f"An error occurred during testing: {e}")

if __name__ == "__main__":
    test_digitization()
