package emr.ekg.features;

/** QTc 정상범위 산정에 사용하는 생물학적 성별. 미상 시 남성 기준(보수적, 더 넓은 하한)을 적용한다. */
public enum Sex {
    MALE, FEMALE, UNKNOWN
}
