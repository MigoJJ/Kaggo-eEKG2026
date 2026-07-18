package emr.ekg.app.view;

import emr.ekg.signal.PreprocessedEcg;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.paint.Color;
import javafx.scene.text.Font;
import javafx.scene.text.FontWeight;

import java.util.HashMap;
import java.util.Locale;
import java.util.Map;

/**
 * 표준 임상 12유도 심전도 인쇄 레이아웃(4열×3행 + 하단 리듬 스트립)을 재현하는 캔버스.
 * 각 열은 10초 레코딩을 순차적으로 4등분한 2.5초 구간을 보여준다 — 실제 임상 프린트 관례이며,
 * 표준 게인(25mm/s, 10mm/mV)을 화면 픽셀로 환산해 격자를 그린다.
 */
public final class EcgWaveformCanvas extends Canvas {

    private static final String[][] LAYOUT = {
            {"I", "aVR", "V1", "V4"},
            {"II", "aVL", "V2", "V5"},
            {"III", "aVF", "V3", "V6"},
    };
    private static final String RHYTHM_LEAD = "II";

    private static final double PX_PER_MM = 4.0;
    private static final double PAPER_SPEED_MM_PER_SEC = 25.0;
    private static final double GAIN_MM_PER_MV = 10.0;
    private static final double PX_PER_SEC = PX_PER_MM * PAPER_SPEED_MM_PER_SEC;
    private static final double PX_PER_MV = PX_PER_MM * GAIN_MM_PER_MV;

    private static final double SEGMENT_SECONDS = 2.5;
    private static final double PANEL_WIDTH = SEGMENT_SECONDS * PX_PER_SEC;
    private static final double PANEL_MV_RANGE = 4.0;
    private static final double PANEL_HEIGHT = PANEL_MV_RANGE * PX_PER_MV;
    private static final double RHYTHM_HEIGHT = PANEL_HEIGHT;

    public static final double CANVAS_WIDTH = LAYOUT[0].length * PANEL_WIDTH;
    public static final double CANVAS_HEIGHT = LAYOUT.length * PANEL_HEIGHT + RHYTHM_HEIGHT;

    private static final Color GRID_MINOR = Color.rgb(255, 205, 205);
    private static final Color GRID_MAJOR = Color.rgb(255, 145, 145);
    private static final Color TRACE_COLOR = Color.rgb(15, 15, 15);
    private static final Color LABEL_COLOR = Color.rgb(0, 0, 0);
    private static final Color RHYTHM_LABEL_COLOR = Color.rgb(70, 70, 70);

    public EcgWaveformCanvas() {
        super(CANVAS_WIDTH, CANVAS_HEIGHT);
    }

    public void render(PreprocessedEcg ecg) {
        GraphicsContext gc = getGraphicsContext2D();
        gc.setFill(Color.WHITE);
        gc.fillRect(0, 0, getWidth(), getHeight());

        int fs = ecg.fs();
        int totalSamples = ecg.sampleCount();
        int segmentSamples = totalSamples / LAYOUT[0].length;
        Map<String, double[]> leadByName = indexLeads(ecg);

        for (int row = 0; row < LAYOUT.length; row++) {
            for (int col = 0; col < LAYOUT[row].length; col++) {
                double x0 = col * PANEL_WIDTH;
                double y0 = row * PANEL_HEIGHT;
                String leadName = LAYOUT[row][col];
                drawGrid(gc, x0, y0, PANEL_WIDTH, PANEL_HEIGHT);

                double[] full = leadByName.get(leadName.toUpperCase(Locale.ROOT));
                if (full != null) {
                    int from = col * segmentSamples;
                    int to = Math.min(full.length, from + segmentSamples);
                    drawTrace(gc, full, from, to, fs, x0, y0, PANEL_HEIGHT);
                    drawLabel(gc, leadName, x0, y0, LABEL_COLOR);
                }
            }
        }

        double rhythmY = LAYOUT.length * PANEL_HEIGHT;
        drawGrid(gc, 0, rhythmY, getWidth(), RHYTHM_HEIGHT);
        double[] rhythmLead = leadByName.get(RHYTHM_LEAD.toUpperCase(Locale.ROOT));
        if (rhythmLead != null) {
            drawTrace(gc, rhythmLead, 0, rhythmLead.length, fs, 0, rhythmY, RHYTHM_HEIGHT);
            drawLabel(gc, RHYTHM_LEAD + " (rhythm)", 0, rhythmY, RHYTHM_LABEL_COLOR);
        }
    }

    private static Map<String, double[]> indexLeads(PreprocessedEcg ecg) {
        Map<String, double[]> map = new HashMap<>();
        String[] names = ecg.leadNames();
        for (int i = 0; i < names.length; i++) {
            map.put(names[i].toUpperCase(Locale.ROOT), ecg.samples()[i]);
        }
        return map;
    }

    private static void drawGrid(GraphicsContext gc, double x0, double y0, double w, double h) {
        gc.save();
        gc.beginPath();
        gc.rect(x0, y0, w, h);
        gc.clip();

        gc.setStroke(GRID_MINOR);
        gc.setLineWidth(0.5);
        for (double x = x0; x <= x0 + w; x += PX_PER_MM) {
            gc.strokeLine(x, y0, x, y0 + h);
        }
        for (double y = y0; y <= y0 + h; y += PX_PER_MM) {
            gc.strokeLine(x0, y, x0 + w, y);
        }

        double majorStep = PX_PER_MM * 5;
        gc.setStroke(GRID_MAJOR);
        gc.setLineWidth(1.0);
        for (double x = x0; x <= x0 + w; x += majorStep) {
            gc.strokeLine(x, y0, x, y0 + h);
        }
        for (double y = y0; y <= y0 + h; y += majorStep) {
            gc.strokeLine(x0, y, x0 + w, y);
        }
        gc.restore();
    }

    private static void drawTrace(GraphicsContext gc, double[] signal, int from, int to, int fs,
            double x0, double y0, double panelHeight) {
        gc.save();
        gc.beginPath();
        gc.rect(x0, y0, (to - from) * PX_PER_SEC / fs, panelHeight);
        gc.clip();

        double baselineY = y0 + panelHeight / 2.0;
        gc.setStroke(TRACE_COLOR);
        gc.setLineWidth(1.2);
        gc.beginPath();
        boolean first = true;
        for (int i = from; i < to; i++) {
            double t = (i - from) / (double) fs;
            double x = x0 + t * PX_PER_SEC;
            double y = baselineY - signal[i] * PX_PER_MV;
            if (first) {
                gc.moveTo(x, y);
                first = false;
            } else {
                gc.lineTo(x, y);
            }
        }
        gc.stroke();
        gc.restore();
    }

    private static void drawLabel(GraphicsContext gc, String text, double x0, double y0, Color color) {
        gc.setFill(color);
        gc.setFont(Font.font("Monospace", FontWeight.BOLD, 13));
        gc.fillText(text, x0 + 4, y0 + 14);
    }
}
