package emr.ekg.app.view;

import emr.ekg.features.ClinicalFeature;
import emr.ekg.pipeline.DiagnosticReport;
import emr.ekg.rules.Finding;
import javafx.beans.property.SimpleStringProperty;
import javafx.geometry.Insets;
import javafx.scene.control.Label;
import javafx.scene.control.TableColumn;
import javafx.scene.control.TableRow;
import javafx.scene.control.TableView;
import javafx.scene.layout.VBox;
import javafx.scene.text.Font;
import javafx.scene.text.FontWeight;

import java.util.Comparator;
import java.util.List;

/**
 * 진단 리포트 표시 패널: 응급 소견 배너, 1단계 우선순위 점수(참고용, 자동확정 아님),
 * 소견 목록(CRITICAL 우선 정렬), 20개 임상 피처(정상범위 이탈 강조), Stage3/4 미제공 안내.
 */
public final class ReportPanel extends VBox {

    private final Label criticalBanner = new Label();
    private final Label headerLabel = new Label();
    private final Label triageLabel = new Label();
    private final TableView<Finding> findingsTable = new TableView<>();
    private final TableView<ClinicalFeature> featuresTable = new TableView<>();
    private final Label stageAvailabilityLabel = new Label();

    public ReportPanel() {
        setSpacing(10);
        setPadding(new Insets(12));

        criticalBanner.setStyle("-fx-background-color:#c62828; -fx-text-fill:white; "
                + "-fx-font-weight:bold; -fx-font-size:15px; -fx-padding:8 12;");
        criticalBanner.setMaxWidth(Double.MAX_VALUE);
        criticalBanner.setVisible(false);
        criticalBanner.setManaged(false);

        headerLabel.setFont(Font.font("System", FontWeight.BOLD, 14));
        triageLabel.setFont(Font.font(13));
        stageAvailabilityLabel.setStyle("-fx-text-fill:#888888; -fx-font-size:11px;");

        setupFindingsTable();
        setupFeaturesTable();

        getChildren().addAll(criticalBanner, headerLabel, triageLabel,
                sectionLabel("소견 (Findings)"), findingsTable,
                sectionLabel("임상 피처 (20종)"), featuresTable,
                stageAvailabilityLabel);
    }

    private static Label sectionLabel(String text) {
        Label l = new Label(text);
        l.setFont(Font.font("System", FontWeight.BOLD, 12));
        return l;
    }

    private void setupFindingsTable() {
        TableColumn<Finding, String> severityCol = new TableColumn<>("심각도");
        severityCol.setCellValueFactory(f -> new SimpleStringProperty(f.getValue().severity().name()));
        TableColumn<Finding, String> codeCol = new TableColumn<>("코드");
        codeCol.setCellValueFactory(f -> new SimpleStringProperty(f.getValue().code()));
        TableColumn<Finding, String> labelCol = new TableColumn<>("소견");
        labelCol.setCellValueFactory(f -> new SimpleStringProperty(f.getValue().label()));
        TableColumn<Finding, String> evidenceCol = new TableColumn<>("근거");
        evidenceCol.setCellValueFactory(f -> new SimpleStringProperty(f.getValue().evidence()));

        severityCol.setPrefWidth(80);
        codeCol.setPrefWidth(120);
        labelCol.setPrefWidth(240);
        evidenceCol.setPrefWidth(340);

        findingsTable.getColumns().setAll(List.of(severityCol, codeCol, labelCol, evidenceCol));
        findingsTable.setPrefHeight(160);
        findingsTable.setPlaceholder(new Label("소견 없음"));
        findingsTable.setRowFactory(tv -> new TableRow<>() {
            @Override
            protected void updateItem(Finding item, boolean empty) {
                super.updateItem(item, empty);
                setStyle(!empty && item != null && item.isCritical() ? "-fx-background-color:#ffebee;" : "");
            }
        });
    }

    private void setupFeaturesTable() {
        TableColumn<ClinicalFeature, String> nameCol = new TableColumn<>("피처");
        nameCol.setCellValueFactory(f -> new SimpleStringProperty(f.getValue().name()));
        TableColumn<ClinicalFeature, String> valueCol = new TableColumn<>("값");
        valueCol.setCellValueFactory(f -> new SimpleStringProperty(
                "%.2f %s".formatted(f.getValue().value(), f.getValue().unit())));
        TableColumn<ClinicalFeature, String> rangeCol = new TableColumn<>("정상범위");
        rangeCol.setCellValueFactory(f -> new SimpleStringProperty(formatRange(f.getValue())));
        TableColumn<ClinicalFeature, String> statusCol = new TableColumn<>("상태");
        statusCol.setCellValueFactory(f -> new SimpleStringProperty(
                f.getValue().inRange() ? "정상범위" : "범위 이탈"));

        nameCol.setPrefWidth(170);
        valueCol.setPrefWidth(120);
        rangeCol.setPrefWidth(140);
        statusCol.setPrefWidth(90);

        featuresTable.getColumns().setAll(List.of(nameCol, valueCol, rangeCol, statusCol));
        featuresTable.setPrefHeight(240);
        featuresTable.setRowFactory(tv -> new TableRow<>() {
            @Override
            protected void updateItem(ClinicalFeature item, boolean empty) {
                super.updateItem(item, empty);
                setStyle(!empty && item != null && !item.inRange() ? "-fx-background-color:#fff3e0;" : "");
            }
        });
    }

    private static final double UNBOUNDED_THRESHOLD = 1.0e9;

    private static String formatRange(ClinicalFeature f) {
        boolean noLower = f.normalLow() <= -UNBOUNDED_THRESHOLD;
        boolean noUpper = f.normalHigh() >= UNBOUNDED_THRESHOLD;
        if (noLower && noUpper) {
            return "제한 없음";
        }
        if (noUpper) {
            return "%.2f 이상".formatted(f.normalLow());
        }
        if (noLower) {
            return "%.2f 이하".formatted(f.normalHigh());
        }
        return "%.2f ~ %.2f".formatted(f.normalLow(), f.normalHigh());
    }

    public void render(DiagnosticReport report, boolean signed) {
        boolean critical = report.hasCriticalFinding();
        criticalBanner.setVisible(critical);
        criticalBanner.setManaged(critical);
        criticalBanner.setText("⚠ 응급 소견 있음 — 즉시 확인 필요");

        String signedSuffix = signed ? " [서명완료]" : "";
        headerLabel.setText("레코드: %s (%s) — %s%s"
                .formatted(report.recordId(), report.source(), report.status(), signedSuffix));
        triageLabel.setText("1단계 우선순위 점수 P(NORM)=%.3f — 자동확정 아님, 검토 순서 참고용 (model %s)"
                .formatted(report.normTriageScore(), report.triageModelVersion()));

        findingsTable.getItems().setAll(
                report.findings().stream()
                        .sorted(Comparator.comparing(f -> f.severity().ordinal()))
                        .toList());
        featuresTable.getItems().setAll(report.features());

        StringBuilder notice = new StringBuilder();
        if (!report.beatArrhythmiaAvailable()) {
            notice.append("Stage3(비트 부정맥 검증) 미제공. ");
        }
        if (!report.stIschemiaAvailable()) {
            notice.append("Stage4(ST-T 허혈 정밀) 미제공.");
        }
        stageAvailabilityLabel.setText(notice.toString());
    }
}
