"""Ownership of the ipc:// socket.

ZeroMQ creates the socket with the process umask, so an engine running as
root leaves it root owned and mode 0755. A client running as another user
cannot connect to that -- it gets EAGAIN and times out, with nothing logged
on either side. Frigate 0.19 runs its detectors as a non-root user.
"""
import os
import tempfile
import unittest

from inference_engine.batch import open_ipc_socket


class OpenIpcSocketTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.sock = os.path.join(self.dir.name, "detector.sock")
        self.owner = os.path.join(self.dir.name, "inference.yaml")
        open(self.sock, "w").close()
        open(self.owner, "w").close()
        os.chmod(self.sock, 0o755)
        self.endpoint = f"ipc://{self.sock}"

    def tearDown(self):
        self.dir.cleanup()

    def _mode(self):
        return os.stat(self.sock).st_mode & 0o777

    def test_non_ipc_endpoint_is_ignored(self):
        open_ipc_socket("tcp://127.0.0.1:5555", self.owner)
        self.assertEqual(self._mode(), 0o755)

    def test_root_owned_config_leaves_the_socket_alone(self):
        """Either the client is root, where 0755 already works, or the config
        is not Frigate's. Chowning to root would silently recreate the bug."""
        if os.stat(self.owner).st_uid != 0:
            self.skipTest("test does not run as root")
        open_ipc_socket(self.endpoint, self.owner)
        self.assertEqual(self._mode(), 0o755)
        self.assertEqual(os.stat(self.sock).st_uid, 0)

    def test_missing_config_leaves_the_socket_alone(self):
        open_ipc_socket(self.endpoint, os.path.join(self.dir.name, "absent.yaml"))
        self.assertEqual(self._mode(), 0o755)

    def test_explicit_uid_is_applied(self):
        if os.geteuid() != 0:
            self.skipTest("chown requires root")
        open_ipc_socket(self.endpoint, None, uid=1000, gid=1000)
        st = os.stat(self.sock)
        self.assertEqual((st.st_uid, st.st_gid), (1000, 1000))
        self.assertEqual(self._mode(), 0o660)

    def test_config_owner_is_copied_onto_the_socket(self):
        """The whole point: follow Frigate's PUID without being told it."""
        if os.geteuid() != 0:
            self.skipTest("chown requires root")
        os.chown(self.owner, 1000, 1000)
        open_ipc_socket(self.endpoint, self.owner)
        st = os.stat(self.sock)
        self.assertEqual((st.st_uid, st.st_gid), (1000, 1000))
        self.assertEqual(self._mode(), 0o660)


if __name__ == "__main__":
    unittest.main()
