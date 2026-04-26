import cv2
import numpy as np
import matplotlib.pyplot as plt
import fitz  # PyMuPDF
import os

def convert_pdf_to_images(pdf_path, dpi=300):
    """
    PDF 파일의 각 페이지를 고해상도 이미지로 변환합니다.
    """
    doc = fitz.open(pdf_path)
    images = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        if pix.n == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        elif pix.n == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        images.append(img_array)
    doc.close()
    return images

def digitize_ecg_data(img):
    """
    이미지 배열(numpy)로부터 ECG 파형을 수치화합니다.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
    img_no_grid = img.copy()
    img_no_grid[mask > 0] = [255, 255, 255]
    gray = cv2.cvtColor(img_no_grid, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    height, width = binary.shape
    signal = []
    for x in range(width):
        black_pixels = np.where(binary[:, x] > 0)[0]
        if len(black_pixels) > 0:
            signal.append(np.mean(black_pixels))
        else:
            signal.append(signal[-1] if signal else height/2)
    if len(signal) == 0: return np.zeros(1000)
    signal = np.max(signal) - np.array(signal)
    return signal

def segment_leads(img, rows=4, cols=3):
    """
    ECG 이미지를 격자 단위로 분할하여 12개 리드 영역을 반환합니다.
    """
    h, w = img.shape[:2]
    lead_h, lead_w = h // rows, w // cols
    leads = []
    for r in range(rows):
        for c in range(cols):
            lead_img = img[r*lead_h:(r+1)*lead_h, c*lead_w:(c+1)*lead_w]
            leads.append(lead_img)
    return leads

def process_ecg_document(file_path):
    """
    이미지 또는 PDF 파일을 입력받아 12개 리드 신호를 개별 추출합니다.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        images = convert_pdf_to_images(file_path)
        img = images[0]
    elif ext in ['.jpg', '.jpeg', '.png']:
        img = cv2.imread(file_path)
    else:
        raise ValueError("지원하지 않는 파일 형식입니다.")

    lead_images = segment_leads(img)
    all_lead_signals = []
    for i, l_img in enumerate(lead_images):
        sig = digitize_ecg_data(l_img)
        if len(sig) > 1000:
            sig = sig[:1000]
        else:
            sig = np.pad(sig, (0, 1000 - len(sig)), mode='edge')
        all_lead_signals.append(sig)
    return np.array(all_lead_signals).T

if __name__ == "__main__":
    print("ECG Document Digitizer Loaded.")
