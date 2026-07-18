package emr.ekg.signal.wfdb;

/** WFDB format 16: 16비트 signed little-endian, 리드 인터리브(PTB-XL 표준). */
final class Format16Codec {

    private Format16Codec() {
    }

    static int[] decode(byte[] data, int totalSamples) {
        int[] out = new int[totalSamples];
        for (int i = 0; i < totalSamples; i++) {
            int lo = data[2 * i] & 0xFF;
            int hi = data[2 * i + 1] & 0xFF;
            out[i] = (short) ((hi << 8) | lo);
        }
        return out;
    }
}
