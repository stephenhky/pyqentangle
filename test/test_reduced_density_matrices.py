
import numpy as np
import pytest

import pyqentangle
import pyqentangle.tncompute


def test_two_level():
    tensor = np.array([[0., np.sqrt(0.6) * 1j], [np.sqrt(0.4) * 1j, 0.]])

    # subsystem A
    reddenmat0 = pyqentangle.tncompute.bipartitepurestate_reduceddensitymatrix(tensor, 0)
    assert np.trace(reddenmat0), pytest.approx(1.0)
    assert reddenmat0[0, 0], pytest.approx(0.6)
    assert reddenmat0[1, 1], pytest.approx(0.4)
    assert reddenmat0[0, 1], pytest.approx(0.+0.j)
    assert reddenmat0[1, 0], pytest.approx(0.+0.j)

    # subsystem B
    reddenmat1 = pyqentangle.tncompute.bipartitepurestate_reduceddensitymatrix(tensor, 1)
    assert np.trace(reddenmat1), pytest.approx(1.)
    assert reddenmat1[0, 0], pytest.approx(0.4)
    assert reddenmat1[1, 1], pytest.approx(0.6)
    assert reddenmat1[0, 1], pytest.approx(0.+0.j)
    assert reddenmat1[1, 0], pytest.approx(0.+0.j)

def test_three_level():
    tensor = np.array([[np.sqrt(0.5), 0.0, 0.0], [0.0, np.sqrt(0.3)*1.j, 0.0], [0.0, 0.0, np.sqrt(0.2)]])

    # subsystem A
    reddenmat0 = pyqentangle.tncompute.bipartitepurestate_reduceddensitymatrix(tensor, 0)
    assert np.trace(reddenmat0), pytest.approx(1.0)
    assert reddenmat0[0, 0], pytest.approx(0.5)
    assert reddenmat0[1, 1], pytest.approx(0.3)
    assert reddenmat0[2, 2], pytest.approx(0.2)
    assert reddenmat0[1, 0], pytest.approx(0.+0.j)
    assert reddenmat0[1, 2], pytest.approx(0.+0.j)
    assert reddenmat0[2, 0], pytest.approx(0.+0.j)

    # subsystem B
    reddenmat1 = pyqentangle.tncompute.bipartitepurestate_reduceddensitymatrix(tensor, 1)
    assert np.trace(reddenmat1), pytest.approx(1.0)
    assert reddenmat1[0, 0], pytest.approx(0.5)
    assert reddenmat1[1, 1], pytest.approx(0.3)
    assert reddenmat1[2, 2], pytest.approx(0.2)
    assert reddenmat1[2, 1], pytest.approx(0.+0.j)
    assert reddenmat1[1, 0], pytest.approx(0.+0.j)
    assert reddenmat1[0, 2], pytest.approx(0.+0.j)

def test_two_levels_2():
    tensor = np.array([[np.sqrt(0.5), np.sqrt(0.25) * 1.j], [0., np.sqrt(0.25)]])

    # subsystem A
    reddenmat0 = pyqentangle.tncompute.bipartitepurestate_reduceddensitymatrix(tensor, 0)
    assert np.trace(reddenmat0), pytest.approx(1.0)
    assert reddenmat0[0, 0], pytest.approx(0.75)
    assert reddenmat0[1, 1], pytest.approx(0.25)
    assert reddenmat0[0, 1], pytest.approx(0.+0.25j)
    assert reddenmat0[1, 0], pytest.approx(0.-0.25j)

    # subsystem B
    reddenmat1 = pyqentangle.tncompute.bipartitepurestate_reduceddensitymatrix(tensor, 1)
    assert np.trace(reddenmat1), pytest.approx(1.)
    assert reddenmat1[0, 0], pytest.approx(0.5)
    assert reddenmat1[1, 1], pytest.approx(0.5)
    assert reddenmat1[0, 1], pytest.approx(0.-0.35355339j)
    assert reddenmat1[1, 0], pytest.approx(0.+0.35355339j)


def test_fifteen_levels():
    tensor = np.zeros((15, 15))
    tensor[1, 1] = np.sqrt(0.3)
    tensor[1, 2] = np.sqrt(0.1)
    tensor[1, 3] = np.sqrt(0.1)
    tensor[11, 10] = np.sqrt(0.2)
    tensor[11, 11] = np.sqrt(0.1)
    tensor[11, 12] = np.sqrt(0.2)
    reddenmat0 = pyqentangle.tncompute.bipartitepurestate_reduceddensitymatrix(tensor, 0)
    reddenmat1 = pyqentangle.tncompute.bipartitepurestate_reduceddensitymatrix(tensor, 0)
    assert np.trace(reddenmat0), pytest.approx(1.0)
    assert np.trace(reddenmat1), pytest.approx(1.0)
