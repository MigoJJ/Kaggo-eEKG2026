package emr.ekg.rules;

import java.util.List;

/** STEMI 판정에 필요한 인접(contiguous) 리드 그룹 (AHA/ACC/ESC 영역별 분류). */
final class LeadGroups {

    static final List<String> ANTERIOR = List.of("V1", "V2", "V3", "V4");
    static final List<String> INFERIOR = List.of("II", "III", "AVF");
    static final List<String> LATERAL = List.of("I", "AVL", "V5", "V6");

    private LeadGroups() {
    }
}
