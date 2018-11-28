import pytest
from os.path import abspath, basename, dirname, join
import subprocess, glob, sys

cwd = abspath(dirname(__file__))
demo_dir = join(cwd, "../..", "demo/undocumented")

@pytest.fixture(params=glob.glob("%s/*/*.py" % demo_dir),
                ids=lambda x: basename(dirname(x)) + "/" + basename(x))
def demo_file(request):
    return abspath(request.param)

def test_demo_runs(demo_file):
    subprocess.check_call([sys.executable,demo_file])
