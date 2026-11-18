import hashlib
from pathlib import Path
import tempfile
import unittest

from sober.checkpoint import file_sha256, load_checkpoint
from sober.evaluation import verify_example_files


class CheckpointVerificationTests(unittest.TestCase):
    def test_wrong_file_is_rejected_before_deserialization(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pth'
            path.write_bytes(b'not the released weights')
            self.assertEqual(file_sha256(path), hashlib.sha256(path.read_bytes()).hexdigest())
            with self.assertRaisesRegex(ValueError, 'SHA-256'):
                load_checkpoint(path, expected_sha256='0' * 64)

    def test_modified_example_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / 'study' / 'region'
            folder.mkdir(parents=True)
            path = folder / 'expression.csv'
            path.write_text('cell_id,CD3e\n1,2\n')
            records = [{'folder': 'study/region', 'files': {
                'expression.csv': {'sha256': file_sha256(path)},
            }}]
            verify_example_files(root, records)
            path.write_text('cell_id,CD3e\n1,20\n')
            with self.assertRaisesRegex(ValueError, 'SHA-256'):
                verify_example_files(root, records)
