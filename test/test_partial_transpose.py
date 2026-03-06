
import numpy as np
import pytest

from pyqentangle.core.tncompute import bipartitepurestate_densitymatrix, \
    bipartitepurestate_partialtranspose_densitymatrix, flatten_bipartite_densitymatrix


tensor = np.array([[0., np.sqrt(0.6) * 1j], [np.sqrt(0.4) * 1j, 0.]])


def test_partialtranpose():
    fullden = bipartitepurestate_densitymatrix(tensor)
    fullden_pt0 = bipartitepurestate_partialtranspose_densitymatrix(tensor, 0)
    fullden_pt1 = bipartitepurestate_partialtranspose_densitymatrix(tensor, 1)

    assert fullden[0, 1, 0, 1] == pytest.approx(0.6+0j)
    assert fullden[0, 1, 1, 0] == pytest.approx(np.sqrt(0.6*0.4)+0j)
    assert fullden[1, 0, 0, 1] == pytest.approx(np.sqrt(0.6*0.4)+0j)
    assert fullden[1, 0, 1, 0] == pytest.approx(0.4+0j)

    assert fullden_pt0[0, 1, 0, 1] == pytest.approx(0.6+0j)
    assert fullden_pt0[1, 1, 0, 0] == pytest.approx(np.sqrt(0.6*0.4)+0j)
    assert fullden_pt0[0, 0, 1, 1] == pytest.approx(np.sqrt(0.6*0.4)+0j)
    assert fullden_pt0[1, 0, 1, 0] == pytest.approx(0.4+0j)

    assert fullden_pt1[0, 1, 0, 1] == pytest.approx(0.6+0j)
    assert fullden_pt1[0, 0, 1, 1] == pytest.approx(np.sqrt(0.6*0.4)+0j)
    assert fullden_pt1[1, 1, 0, 0] == pytest.approx(np.sqrt(0.6*0.4)+0j)
    assert fullden_pt1[1, 0, 1, 0] == pytest.approx(0.4+0j)


def test_flatten():
    fullden = bipartitepurestate_densitymatrix(tensor)
    fullden_pt0 = bipartitepurestate_partialtranspose_densitymatrix(tensor, 0)
    fullden_pt1 = bipartitepurestate_partialtranspose_densitymatrix(tensor, 1)

    flatten_fullden = flatten_bipartite_densitymatrix(fullden)
    flatten_fullden_pt0 = flatten_bipartite_densitymatrix(fullden_pt0)
    flatten_fullden_pt1 = flatten_bipartite_densitymatrix(fullden_pt1)

    assert flatten_fullden[1, 1] == pytest.approx(0.6+0j)
    assert flatten_fullden[1, 2] == pytest.approx(np.sqrt(0.6*0.4)+0j)
    assert flatten_fullden[2, 1] == pytest.approx(np.sqrt(0.6*0.4)+0j)
    assert flatten_fullden[2, 2] == pytest.approx(0.4+0j)

    assert flatten_fullden_pt0[1, 1] == pytest.approx(0.6+0j)
    assert flatten_fullden_pt0[3, 0] == pytest.approx(np.sqrt(0.6*0.4)+0j)
    assert flatten_fullden_pt0[0, 3] == pytest.approx(np.sqrt(0.6*0.4)+0j)
    assert flatten_fullden_pt0[2, 2] == pytest.approx(0.4+0j)

    assert flatten_fullden_pt1[1, 1] == pytest.approx(0.6+0j)
    assert flatten_fullden_pt1[0, 3] == pytest.approx(np.sqrt(0.6*0.4)+0j)
    assert flatten_fullden_pt1[3, 0] == pytest.approx(np.sqrt(0.6*0.4)+0j)
    assert flatten_fullden_pt1[2, 2] == pytest.approx(0.4+0j)
