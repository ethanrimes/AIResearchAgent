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

def test_data_dir_unwritable():
    data_dir = "/app/data"
    assert os.path.isdir(data_dir)
    # writing should be forbidden
    fn = os.path.join(data_dir, "should_fail.txt")
    with pytest.raises((PermissionError, OSError)):
        with open(fn, "w") as f:
            f.write("nope")

def test_writable_parent_unwritable():
    # The parent writable directory should be read-only
    parent_dir = "/app/sandbox/writable"
    assert os.path.isdir(parent_dir)
    # writing should be forbidden
    fn = os.path.join(parent_dir, "should_fail.txt")
    with pytest.raises((PermissionError, OSError)):
        with open(fn, "w") as f:
            f.write("nope")

def test_run_dir_writable():
    # Get the specific run directory from environment variable
    run_dir = os.environ.get("EXPERIMENT_RUN_PATH")
    assert run_dir, "EXPERIMENT_RUN_PATH environment variable not set"
    assert os.path.isdir(run_dir), f"Run directory {run_dir} does not exist"
    
    # Test writing to subdirectories
    for sub in ("planning", "experiment", "results", "scratchpad"):
        path = os.path.join(run_dir, sub)
        assert os.path.isdir(path), f"Subdirectory {path} does not exist"
        
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