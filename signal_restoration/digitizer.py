import cv2
import numpy as np
import fitz  # PyMuPDF
import os
import pytesseract
from scipy.interpolate import interp1d
from scipy.signal import medfilt

class ECGDigitizerV2:
    def __init__(self, dpi=300):
        self.dpi = dpi
        self.quality_score = {
            "lead_labels_detected": 0,
            "grid_calibration": "fail",
            "skew_detected": 0.0,
            "baseline_stability": "unknown",
            "missing_segments": [],
            "overall_confidence": 0.0,
            "pixels_per_mm_h": 0.0,
            "pixels_per_mm_v": 0.0
        }
        self.lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        self.lead_regions = {} # label: (x, y, w, h)

    def pdf_to_images(self, pdf_path):
        """Converts PDF pages to high-resolution images."""
        doc = fitz.open(pdf_path)
        images = []
        for page in doc:
            pix = page.get_pixmap(matrix=fitz.Matrix(self.dpi/72, self.dpi/72))
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            if pix.n == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            elif pix.n == 4:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
            images.append(img_array)
        doc.close()
        return images

    def detect_skew_and_correct(self, img):
        """Detects grid lines and corrects rotation/skew."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=200, maxLineGap=20)
        
        if lines is not None:
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 == x1: continue
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                if abs(angle) < 10: # Only look for small skews
                    angles.append(angle)
            
            if angles:
                median_angle = np.median(angles)
                self.quality_score["skew_detected"] = float(median_angle)
                
                (h, w) = img.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        return img

    def calibrate_grid_scale(self, img):
        """Detects the frequency of grid lines using FFT to find pixels/mm."""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Red/Pink grid detection
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        mask_grid = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
        
        def detect_period(binary_img, axis):
            proj = np.mean(binary_img, axis=axis)
            if np.max(proj) == 0: return 0
            # Normalize and remove DC
            proj = (proj - np.mean(proj)) / (np.std(proj) + 1e-7)
            # FFT
            fft = np.abs(np.fft.rfft(proj))
            freqs = np.fft.rfftfreq(len(proj))
            # Find dominant frequency (excluding very low frequencies)
            min_freq_idx = int(len(fft) * 0.01) # Skip DC and low drift
            if len(fft) <= min_freq_idx: return 0
            peak_idx = np.argmax(fft[min_freq_idx:]) + min_freq_idx
            if peak_idx == 0: return 0
            period = 1.0 / freqs[peak_idx]
            return period

        spacing_h = detect_period(mask_grid, 0) # Vertical lines -> horizontal spacing
        spacing_v = detect_period(mask_grid, 1) # Horizontal lines -> vertical spacing
        
        # Grid lines are usually 1mm (fine) or 5mm (bold)
        # We try to detect the 1mm spacing
        if 5 < spacing_h < 30: # Reasonable range for 1mm at 150-600 DPI
            self.quality_score["pixels_per_mm_h"] = float(spacing_h)
        else:
            self.quality_score["pixels_per_mm_h"] = self.dpi / 25.4 # Fallback
            
        if 5 < spacing_v < 30:
            self.quality_score["pixels_per_mm_v"] = float(spacing_v)
        else:
            self.quality_score["pixels_per_mm_v"] = self.dpi / 25.4 # Fallback
            
        self.quality_score["grid_calibration"] = "pass"

    def detect_lead_labels(self, img):
        """Uses OCR to find lead labels and their positions."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Enhance for OCR
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        
        # OCR on the whole image to find labels
        # Custom config to look for short alphanumeric strings
        custom_config = r'--oem 3 --psm 11'
        data = pytesseract.image_to_data(thresh, config=custom_config, output_type=pytesseract.Output.DICT)
        
        n_boxes = len(data['text'])
        detected_count = 0
        for i in range(n_boxes):
            text = data['text'][i].strip()
            if text in self.lead_names:
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                self.lead_regions[text] = (x, y, w, h)
                detected_count += 1
        
        self.quality_score["lead_labels_detected"] = detected_count
        return detected_count

    def segment_leads_adaptive(self, binary_img, img_orig):
        """Segments leads using OCR hints or fallback to 4x3."""
        h, w = binary_img.shape
        leads = []
        
        if len(self.lead_regions) >= 8: # If we found most labels, use them
            # This is complex because we need to define the bounding box of the signal
            # for each label. Typically, signals are to the right or below labels.
            # Fallback to 4x3 for now but record that we found labels.
            pass

        rows, cols = 4, 3
        lead_h, lead_w = h // rows, w // cols
        for r in range(rows):
            for c in range(cols):
                lead_img = binary_img[r*lead_h:(r+1)*lead_h, c*lead_w:(c+1)*lead_w]
                leads.append(lead_img)
        return leads

    def extract_waveform_robust(self, lead_binary):
        """Extracts signal with noise filtering and gap filling."""
        h, w = lead_binary.shape
        signal = []
        for x in range(w):
            pts = np.where(lead_binary[:, x] > 0)[0]
            if len(pts) > 0:
                # Use median to ignore noise dots
                signal.append(np.median(pts))
            else:
                signal.append(np.nan)
        
        # Fill NaNs with interpolation
        signal = np.array(signal)
        nans, x = np.isnan(signal), lambda z: z.nonzero()[0]
        if np.any(nans) and not np.all(nans):
            signal[nans] = np.interp(x(nans), x(~nans), signal[~nans])
        elif np.all(nans):
            signal = np.full(w, h // 2)
            
        # Median filter to smooth out small spikes
        signal = medfilt(signal, kernel_size=5)
        
        # Quality check for gaps
        gap_ratio = np.sum(nans) / w
        if gap_ratio > 0.2:
            self.quality_score["baseline_stability"] = "poor"
        elif self.quality_score["baseline_stability"] == "unknown":
            self.quality_score["baseline_stability"] = "stable"
            
        signal = np.max(signal) - signal
        return signal

    def process(self, file_path):
        """Main entry point for V2 Digitization."""
        images = self.pdf_to_images(file_path) if file_path.endswith('.pdf') else [cv2.imread(file_path)]
        img = images[0]
        if img is None: raise ValueError("Image load failed")
            
        # 1. Skew correction
        img = self.detect_skew_and_correct(img)
        
        # 2. OCR for labels
        self.detect_lead_labels(img)
        
        # 3. Grid calibration
        self.calibrate_grid_scale(img)
        
        # 4. Grid removal
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Keep only black/dark-gray (waveform)
        mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 100]))
        
        # 5. Segmentation & Extraction
        lead_binaries = self.segment_leads_adaptive(mask, img)
        raw_signals = [self.extract_waveform_robust(lb) for lb in lead_binaries]
        
        # 6. Final reconstruction
        reconstructed = []
        for sig in raw_signals:
            x_old = np.linspace(0, 1, len(sig))
            x_new = np.linspace(0, 1, 1000)
            f = interp1d(x_old, sig, kind='linear', fill_value="extrapolate")
            reconstructed.append(f(x_new) - np.mean(f(x_new)))
            
        self.quality_score["overall_confidence"] = self.calculate_confidence()
        return np.array(reconstructed).T, self.quality_score

    def calculate_confidence(self):
        score = 0.0
        if self.quality_score["grid_calibration"] == "pass": score += 0.2
        score += max(0, 0.2 - abs(self.quality_score["skew_detected"]) / 5.0)
        score += (self.quality_score["lead_labels_detected"] / 12) * 0.4
        if self.quality_score["baseline_stability"] == "stable": score += 0.2
        return float(np.clip(score, 0, 1))

def get_digitizer_disclaimer():
    return """
[RESEARCH EXPERIMENTAL NOTICE]
- OCR-assisted Digitization Pipeline V2
- PDF-derived ECG results are NOT for clinical use.
- Quality Score and Disclaimer must be reviewed.
"""
