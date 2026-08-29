package emr.ekg.app;

import emr.ekg.app.view.EcgWaveformCanvas;
import emr.ekg.app.view.ReportPanel;
import emr.ekg.features.Sex;
import emr.ekg.persistence.ReportStatus;
import emr.ekg.pipeline.DiagnosticReport;
import emr.ekg.pipeline.PipelineResult;
import javafx.animation.PauseTransition;
import javafx.application.Application;
import javafx.beans.property.SimpleStringProperty;
import javafx.embed.swing.SwingFXUtils;
import javafx.geometry.Insets;
import javafx.scene.Scene;
import javafx.scene.SnapshotParameters;
import javafx.scene.control.Alert;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.ScrollPane;
import javafx.scene.control.Tab;
import javafx.scene.control.TabPane;
import javafx.scene.control.TableColumn;
import javafx.scene.control.TableRow;
import javafx.scene.control.TableView;
import javafx.scene.control.TextField;
import javafx.scene.control.ToolBar;
import javafx.scene.image.WritableImage;
import javafx.scene.layout.BorderPane;
import javafx.stage.Stage;
import javafx.util.Duration;

import javax.imageio.ImageIO;
import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

/**
 * EKG-GDS EMR 2026 JavaFX 진입점.
 *
 * 좌측 서명 큐 → 레코드 선택 → 중앙 탭(표준 12유도 파형 / 진단 리포트) → 서명, 의 실제 임상
 * 판독 보조 워크플로를 구현한다. Stage1 우선순위 점수는 게이팅에 쓰이지 않으며(P3),
 * 모든 리포트는 의사 서명이 필요하다.
 */
public final class EmrApp extends Application {

    private static final Path PTBXL_ROOT = Path.of(
            "/mnt/t7/datasets/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3");
    private static final Path CONFIG_YAML = Path.of("/home/migojj/ittia/Kaggo-eEKG2026/config/pipeline.yaml");
    private static final Path DB_FILE = Path.of("/home/migojj/ittia/Kaggo-eEKG2026/data/emr.db");
    // ecg_id 1,3,4: PTB-XL scp_codes NORM=100.0 (확인된 순수 정상 라벨). 1000도 NORM=100.0이나
    // 1도AVB/LAD 룰소견이 함께 나와 데모 다양성을 위해 포함(정상 라벨이어도 경계성 룰소견은
    // 발생할 수 있음을 보여줌).
    private static final int[] DEMO_ECG_IDS = {1, 3, 4, 1000};

    private final TableView<PipelineResult> queueTable = new TableView<>();
    private final EcgWaveformCanvas waveformCanvas = new EcgWaveformCanvas();
    private final ReportPanel reportPanel = new ReportPanel();
    private final Label statusBar = new Label("준비됨");
    private final Set<String> signedIds = new HashSet<>();
    private TabPane centerTabs;

    private AppContext appContext;
    private PipelineResult selected;

    @Override
    public void start(Stage stage) throws Exception {
        appContext = new AppContext(CONFIG_YAML, DB_FILE);

        BorderPane root = new BorderPane();
        root.setTop(buildToolbar());
        root.setLeft(buildQueuePanel());
        root.setCenter(buildCenterTabs());
        root.setBottom(statusBar);
        BorderPane.setMargin(statusBar, new Insets(4, 8, 4, 8));

        Scene scene = new Scene(root, 1300, 820);
        stage.setScene(scene);
        stage.setTitle("EKG-GDS EMR 2026 — 심전도 판독 보조 시스템");
        stage.show();

        loadDemoRecords();

        String debugScreenshotPath = System.getProperty("ekg.debug.screenshot");
        if (debugScreenshotPath != null) {
            PauseTransition delay = new PauseTransition(Duration.millis(600));
            delay.setOnFinished(e -> {
                captureDebugScreenshot(root, Path.of(debugScreenshotPath));
                centerTabs.getSelectionModel().select(1);
                PauseTransition reportDelay = new PauseTransition(Duration.millis(300));
                reportDelay.setOnFinished(e2 -> captureDebugScreenshot(root,
                        Path.of(debugScreenshotPath.replace(".png", "_report.png"))));
                reportDelay.play();
            });
            delay.play();
        }
    }

    /** 시각 검증용 디버그 스냅샷 — X11/Wayland 화면캡처 도구에 의존하지 않고 JavaFX 씬을 직접 PNG로 렌더링한다. */
    private void captureDebugScreenshot(javafx.scene.Node node, Path outPath) {
        try {
            WritableImage image = node.snapshot(new SnapshotParameters(), null);
            ImageIO.write(SwingFXUtils.fromFXImage(image, null), "png", new File(outPath.toString()));
            System.out.println("DEBUG_SCREENSHOT_SAVED: " + outPath);
        } catch (IOException e) {
            System.err.println("DEBUG_SCREENSHOT_FAILED: " + e.getMessage());
        }
    }

    private ToolBar buildToolbar() {
        TextField ecgIdField = new TextField();
        ecgIdField.setPromptText("PTB-XL ecg_id (예: 5000)");
        ecgIdField.setPrefWidth(160);

        Button loadButton = new Button("레코드 불러오기");
        loadButton.setOnAction(e -> {
            String text = ecgIdField.getText().trim();
            if (text.isEmpty()) {
                return;
            }
            try {
                loadPtbxlRecord(Integer.parseInt(text));
            } catch (NumberFormatException ex) {
                showError("ecg_id는 숫자여야 합니다: " + text);
            }
        });

        Button signButton = new Button("서명");
        signButton.setOnAction(e -> signSelected());

        return new ToolBar(new Label("ecg_id:"), ecgIdField, loadButton, signButton);
    }

    private BorderPane buildQueuePanel() {
        TableColumn<PipelineResult, String> idCol = new TableColumn<>("Record");
        idCol.setCellValueFactory(f -> new SimpleStringProperty(f.getValue().report().recordId()));
        TableColumn<PipelineResult, String> statusCol = new TableColumn<>("상태");
        statusCol.setCellValueFactory(f -> {
            DiagnosticReport r = f.getValue().report();
            boolean signed = signedIds.contains(r.recordId());
            String label = signed ? "SIGNED" : r.status().name();
            return new SimpleStringProperty(label);
        });
        TableColumn<PipelineResult, String> triageCol = new TableColumn<>("P(NORM)");
        triageCol.setCellValueFactory(f -> new SimpleStringProperty(
                String.format(Locale.ROOT, "%.3f", f.getValue().report().normTriageScore())));
        TableColumn<PipelineResult, String> criticalCol = new TableColumn<>("응급");
        criticalCol.setCellValueFactory(f -> new SimpleStringProperty(
                f.getValue().report().hasCriticalFinding() ? "⚠ CRITICAL" : ""));

        idCol.setPrefWidth(90);
        statusCol.setPrefWidth(90);
        triageCol.setPrefWidth(80);
        criticalCol.setPrefWidth(90);

        queueTable.getColumns().setAll(List.of(idCol, statusCol, triageCol, criticalCol));
        queueTable.setPrefWidth(360);
        queueTable.setRowFactory(tv -> new TableRow<>() {
            @Override
            protected void updateItem(PipelineResult item, boolean empty) {
                super.updateItem(item, empty);
                setStyle(!empty && item != null && item.report().hasCriticalFinding()
                        ? "-fx-background-color:#ffebee;" : "");
            }
        });
        queueTable.getSelectionModel().selectedItemProperty().addListener((obs, old, next) -> {
            if (next != null) {
                select(next);
            }
        });

        BorderPane pane = new BorderPane(queueTable);
        pane.setTop(new Label("서명 대기 큐"));
        pane.setPadding(new Insets(8));
        return pane;
    }

    private TabPane buildCenterTabs() {
        ScrollPane waveformScroll = new ScrollPane(waveformCanvas);
        waveformScroll.setPannable(true);

        Tab waveformTab = new Tab("표준 12유도 파형", waveformScroll);
        waveformTab.setClosable(false);
        Tab reportTab = new Tab("진단 리포트", reportPanel);
        reportTab.setClosable(false);

        centerTabs = new TabPane(waveformTab, reportTab);
        return centerTabs;
    }

    private void loadDemoRecords() {
        for (int ecgId : DEMO_ECG_IDS) {
            loadPtbxlRecord(ecgId);
        }
        if (!queueTable.getItems().isEmpty()) {
            queueTable.getSelectionModel().selectFirst();
        }
    }

    private void loadPtbxlRecord(int ecgId) {
        try {
            String paddedId = String.format(Locale.ROOT, "%05d", ecgId);
            String dir = String.format(Locale.ROOT, "%05d", (ecgId / 1000) * 1000);
            Path header = PTBXL_ROOT.resolve("records500").resolve(dir).resolve(paddedId + "_hr.hea");

            PipelineResult result = appContext.loadWfdbRecord(
                    "ptbxl-" + ecgId, "ptbxl:" + ecgId, header, Sex.UNKNOWN);
            queueTable.getItems().add(result);
            queueTable.getSelectionModel().select(result);
            statusBar.setText("불러옴: ecg_id=" + ecgId + " status=" + result.report().status());
        } catch (Exception e) {
            showError("레코드 불러오기 실패 (ecg_id=" + ecgId + "): " + e.getMessage());
        }
    }

    private void select(PipelineResult result) {
        this.selected = result;
        waveformCanvas.render(result.ecg());
        reportPanel.render(result.report(), signedIds.contains(result.report().recordId()));
    }

    private void signSelected() {
        if (selected == null) {
            return;
        }
        String recordId = selected.report().recordId();
        if (selected.report().status() != ReportStatus.PENDING_SIGN) {
            showError("서명 대기 상태가 아닌 리포트는 서명할 수 없습니다: " + selected.report().status());
            return;
        }
        try {
            appContext.sign(recordId, "dr.demo");
            signedIds.add(recordId);
            queueTable.refresh();
            reportPanel.render(selected.report(), true);
            statusBar.setText("서명 완료: " + recordId);
        } catch (Exception e) {
            showError("서명 실패: " + e.getMessage());
        }
    }

    private void showError(String message) {
        statusBar.setText("오류: " + message);
        Alert alert = new Alert(Alert.AlertType.ERROR, message);
        alert.showAndWait();
    }

    public static void main(String[] args) {
        launch(args);
    }
}
