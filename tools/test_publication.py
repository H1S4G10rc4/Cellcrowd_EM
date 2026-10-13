import unittest

from check_publication import allowed_path, check_snapshot


class PublicationTests(unittest.TestCase):
    def check(self, files, modes=None):
        modes = modes or {p: "100644" for p in files}
        return check_snapshot(modes, files.__getitem__)

    def test_source_snapshot(self):
        files = {"PUBLIC_FILES.txt": b"PUBLIC_FILES.txt\nREADME.md\n", "README.md": b"Source only.\n"}
        self.assertEqual(self.check(files), [])

    def test_force_added_data(self):
        files = {"PUBLIC_FILES.txt": b"PUBLIC_FILES.txt\n", "outputs/participants.csv": b"private"}
        self.assertIn("file_set_differs_from_public_allowlist", self.check(files))
        self.assertIn("data_or_unsupported_path", self.check(files))

    def test_disallowed_even_if_listed(self):
        for path in ("private_data/records.py", "data/record.txt", "figure.png", "run.ipynb", "bundle.zip"):
            self.assertFalse(allowed_path(path))

    def test_symlink(self):
        files = {"PUBLIC_FILES.txt": b"PUBLIC_FILES.txt\nlink.py\n", "link.py": b"../private"}
        self.assertIn("link_or_unsupported_file_mode", self.check(files, {"PUBLIC_FILES.txt": "100644", "link.py": "120000"}))

    def test_credential_and_personal_path(self):
        for content in (b"gh" + b"p_" + b"x" * 36, b"/" + b"home/participant/records",
                        b"person" + b"@" + b"example.org"):
            files = {"PUBLIC_FILES.txt": b"PUBLIC_FILES.txt\nREADME.md\n", "README.md": content}
            self.assertIn("possible_credential_or_personal_path", self.check(files))

    def test_binary_disguised_as_source(self):
        files = {"PUBLIC_FILES.txt": b"PUBLIC_FILES.txt\ncode.py\n", "code.py": b"\x00\x01"}
        self.assertIn("binary_control_characters", self.check(files))

    def test_old_history_without_allowlist(self):
        self.assertEqual(self.check({"README.md": b"old"}), ["missing_public_allowlist"])


if __name__ == "__main__":
    unittest.main()
