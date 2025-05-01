# infra/docker/setup_tests/test_permissions.py
import os
import pytest

def test_read_only_dir_unwritable():
    ro = "/app/sandbox/read_only"
    assert os.path.isdir(ro)
    # writing should be forbidden
    fn = os.path.join(ro, "should_fail.txt")
    with pytest.raises((PermissionError, OSError)):
        with open(fn, "w") as f:
            f.write("nope")

def test_writable_subdirs():
    for sub in ("experiments", "results", "scratchpad"):
        path = f"/app/sandbox/writable/{sub}"
        assert os.path.isdir(path)
        testfile = os.path.join(path, "perm_test.txt")
        with open(testfile, "w") as f:
            f.write("ok")
        with open(testfile) as f:
            assert f.read() == "ok"
        os.remove(testfile)

def test_no_global_write():
    # Docker runs as non-root, so writing /root or /etc should fail
    for forbidden in ("/root/foo.txt", "/etc/foo.txt"):
        with pytest.raises((PermissionError, OSError)):
            with open(forbidden, "w") as f:
                f.write("bar")