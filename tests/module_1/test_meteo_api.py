""" This is a dummy example to show how to import code from src/ for testing"""

# from src.module_1.module_1_meteo_api import main

from src.module_1.draft import inc


def test_inc():
    assert inc(4) == 5