
from typing import Literal

import numpy as np
import numpy.typing as npt
import tensornetwork as tn


def schmidt_decomposition_numpy(
        bipartitepurestate_tensor: npt.NDArray[np.complex128]
) -> list[tuple[float, npt.NDArray[np.complex128], npt.NDArray[np.complex128]]]:
    """Compute the Schmidt decomposition of a discrete bipartite pure state using NumPy SVD.

    Called internally by :func:`schmidt_decomposition` when ``approach='numpy'``.

    The input tensor is treated as a matrix whose singular value decomposition yields the
    Schmidt coefficients (singular values) and the eigenmodes of each subsystem (singular
    vectors).  Results are sorted in descending order of Schmidt coefficient.

    Args:
        bipartitepurestate_tensor (numpy.ndarray): 2-D complex array of shape ``(d1, d2)``
            representing a normalised bipartite pure state, where element ``[i, j]``
            is the coefficient of the basis ket :math:`|ij\\rangle`.

    Returns:
        list[tuple[float, numpy.ndarray, numpy.ndarray]]:
            A list of ``min(d1, d2)`` tuples ``(lambda_k, u_k, v_k)`` sorted by
            descending Schmidt coefficient, where:

            * ``lambda_k`` – real Schmidt coefficient (singular value).
            * ``u_k`` – complex 1-D array; eigenmode of the first subsystem.
            * ``v_k`` – complex 1-D array; eigenmode of the second subsystem.
    """
    state_dims = bipartitepurestate_tensor.shape
    mindim = np.min(state_dims)

    vecs1, diags, vecs2_h = np.linalg.svd(bipartitepurestate_tensor)
    vecs2 = vecs2_h.transpose()

    decomposition = [(diags[k], vecs1[:, k], vecs2[:, k])
                     for k in range(mindim)]

    decomposition = sorted(decomposition, key=lambda dec: dec[0], reverse=True)

    return decomposition


def schmidt_decomposition_tensornetwork(
        bipartitepurestate_tensor: npt.NDArray[np.complex128]
) -> list[tuple[float, npt.NDArray[np.complex128], npt.NDArray[np.complex128]]]:
    """Compute the Schmidt decomposition of a discrete bipartite pure state using TensorNetwork SVD.

    Called internally by :func:`schmidt_decomposition` when ``approach='tensornetwork'`` (the
    default).  Wraps the input array in a :class:`tensornetwork.Node` and delegates the SVD to
    :func:`tensornetwork.split_node_full_svd`, then collects and sorts the decomposition terms
    in descending order of Schmidt coefficient.

    Args:
        bipartitepurestate_tensor (numpy.ndarray): 2-D complex array of shape ``(d1, d2)``
            representing a normalised bipartite pure state, where element ``[i, j]``
            is the coefficient of the basis ket :math:`|ij\\rangle`.

    Returns:
        list[tuple[float, numpy.ndarray, numpy.ndarray]]:
            A list of ``min(d1, d2)`` tuples ``(lambda_k, u_k, v_k)`` sorted by
            descending Schmidt coefficient, where:

            * ``lambda_k`` – real Schmidt coefficient (singular value).
            * ``u_k`` – complex 1-D array; eigenmode of the first subsystem.
            * ``v_k`` – complex 1-D array; eigenmode of the second subsystem.
    """
    state_dims = bipartitepurestate_tensor.shape
    mindim = np.min(state_dims)

    node = tn.Node(bipartitepurestate_tensor)
    vecs1, diags, vecs2_h, _ = tn.split_node_full_svd(node, [node[0]], [node[1]])

    decomposition = [(np.real(diags.tensor[k, k]), vecs1.tensor[:, k], vecs2_h.tensor[k, :])
                     for k in range(mindim)]

    decomposition = sorted(decomposition, key=lambda dec: dec[0], reverse=True)

    return decomposition


def schmidt_decomposition(
        bipartitepurestate_tensor: npt.NDArray[np.complex128],
        approach: Literal["tensornetwork", "numpy"] = 'tensornetwork'
) -> list[tuple[float, npt.NDArray[np.complex128], npt.NDArray[np.complex128]]]:
    """Compute the Schmidt decomposition of a discrete bipartite pure state.

    Decomposes the state described by ``bipartitepurestate_tensor`` into Schmidt form:

    .. math::

        |\\Psi\\rangle = \\sum_k \\lambda_k \\, |u_k\\rangle \\otimes |v_k\\rangle

    where :math:`\\lambda_k \\geq 0` are the Schmidt coefficients and
    :math:`\\{|u_k\\rangle\\}`, :math:`\\{|v_k\\rangle\\}` are orthonormal bases for the
    first and second subsystems respectively.

    The decomposition is obtained via singular value decomposition (SVD) of the coefficient
    matrix.  Two backends are supported: ``'numpy'`` (uses :func:`numpy.linalg.svd`) and
    ``'tensornetwork'`` (uses :func:`tensornetwork.split_node_full_svd`).

    Args:
        bipartitepurestate_tensor (numpy.ndarray): 2-D complex array of shape ``(d1, d2)``
            representing a normalised bipartite pure state, where element ``[i, j]``
            is the coefficient of the basis ket :math:`|ij\\rangle`.
        approach (str, optional): Computational backend to use.  Either ``'numpy'`` or
            ``'tensornetwork'``.  Defaults to ``'tensornetwork'``.

    Returns:
        list[tuple[float, numpy.ndarray, numpy.ndarray]]:
            A list of ``min(d1, d2)`` tuples ``(lambda_k, u_k, v_k)`` sorted in descending
            order of Schmidt coefficient, where:

            * ``lambda_k`` – real Schmidt coefficient.
            * ``u_k`` – complex 1-D array; eigenmode of the first subsystem.
            * ``v_k`` – complex 1-D array; eigenmode of the second subsystem.

    Raises:
        ValueError: If ``approach`` is neither ``'numpy'`` nor ``'tensornetwork'``.
    """
    if approach == 'numpy':
        return schmidt_decomposition_numpy(bipartitepurestate_tensor)
    elif approach == 'tensornetwork':
        return schmidt_decomposition_tensornetwork(bipartitepurestate_tensor)
    else:
        raise ValueError(f"Approach is either 'numpy' or 'tensorflow', not {approach}.")
