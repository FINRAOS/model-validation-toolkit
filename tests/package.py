import mvtk


def test_version():
    assert isinstance(mvtk.__version__, str)
