package emr.ekg.signal.wfdb;

/**
 * WFDB format 212: 12비트 two's complement, 2샘플/3바이트 패킹(MIT-BIH 표준).
 * 프레임 인터리브 스트림을 그대로 2개씩 묶어 디코드하므로 리드 수와 무관하게 동작한다.
 */
final class Format212Codec {

    private Format212Codec() {
    }

    static int[] decode(byte[] data, int totalSamples) {
        int[] out = new int[totalSamples];
        int outIdx = 0;
        int byteIdx = 0;
        while (outIdx < totalSamples) {
            int b0 = data[byteIdx] & 0xFF;
            int b1 = data[byteIdx + 1] & 0xFF;
            int b2 = data[byteIdx + 2] & 0xFF;
            out[outIdx++] = signExtend12(((b1 & 0x0F) << 8) | b0);
            if (outIdx < totalSamples) {
                out[outIdx++] = signExtend12(((b1 & 0xF0) << 4) | b2);
            }
            byteIdx += 3;
        }
        return out;
    }

    private static int signExtend12(int v) {
        return (v << 20) >> 20;
    }
}
