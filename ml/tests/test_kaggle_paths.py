import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


ML_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_ROOT))

from ecgml.kaggle_paths import discover_kaggle_roots  # noqa: E402


class KagglePathDiscoveryTest(unittest.TestCase):
    def test_discovers_physionet_mirror_roots(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            ludb = root / "datasets" / "abdessamiguebli" / "qtdb-ludb" / "physionet.org" / "files" / "ludb" / "1.0.1"
            qtdb = root / "datasets" / "abdessamiguebli" / "qtdb-ludb" / "physionet.org" / "files" / "qtdb" / "1.0.0"
            edb = root / "european-st-t-database" / "edb" / "1.0.0"
            ptbxl = root / "ptbxl"
            preprocessed = root / "ptbxl-preprocessed"

            for path in (ludb, qtdb, edb, ptbxl, preprocessed):
                path.mkdir(parents=True)
            (ludb / "RECORDS").write_text("1\n")
            (ludb / "ludb.csv").write_text("id\n")
            (ludb / "1.hea").write_text("1 12 500 5000\n")
            (qtdb / "RECORDS").write_text("sel100\n")
            (qtdb / "sel100.hea").write_text("sel100 2 250 1000\n")
            (edb / "RECORDS").write_text("e0103\n")
            (edb / "e0103.hea").write_text("e0103 2 250 1000\n")
            (ptbxl / "ptbxl_database.csv").write_text("ecg_id,scp_codes,strat_fold\n")
            (preprocessed / "manifest.csv").write_text("ecg_id,fs,interpretable\n")
            (preprocessed / "1.f32").write_bytes(b"0")

            roots = discover_kaggle_roots(root)

            self.assertEqual(ludb, roots.ludb)
            self.assertEqual(qtdb, roots.qtdb)
            self.assertEqual(edb, roots.european_stt)
            self.assertEqual(ptbxl / "ptbxl_database.csv", roots.ptbxl_csv)
            self.assertEqual(preprocessed, roots.ptbxl_preprocessed)


if __name__ == "__main__":
    unittest.main()
