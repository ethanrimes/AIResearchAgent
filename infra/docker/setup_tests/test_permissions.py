# infra/docker/setup_tests/test_permissions.py
import os
import pytest

def test_read_only_dir_unwritable():
    ro = "/read_only"
    assert os.path.isdir(ro)
    # writing should be forbidden
    fn = os.path.join(ro, "should_fail.txt")
    with pytest.raises((PermissionError, OSError)):
        with open(fn, "w") as f:
            f.write("nope")

def test_data_dir_unwritable():
    data_dir = "/data"
    assert os.path.isdir(data_dir)
    # writing should be forbidden
    fn = os.path.join(data_dir, "should_fail.txt")
    with pytest.raises((PermissionError, OSError)):
        with open(fn, "w") as f:
            f.write("nope")

# def test_writable_parent_unwritable():
#     # The parent writable directory should be read-only
#     parent_dir = "/app/sandbox/writable"
#     assert os.path.isdir(parent_dir)
#     # writing should be forbidden
#     fn = os.path.join(parent_dir, "should_fail.txt")
#     with pytest.raises((PermissionError, OSError)):
#         with open(fn, "w") as f:
#             f.write("nope")

def test_run_dir_writable():
    # Test writing to root-level directories
    for sub in ("experiment", "planning", "findings", "results", "hypothesis", "scratchpad"):
        path = f"/{sub}"
        assert os.path.isdir(path), f"Directory {path} does not exist"
        
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