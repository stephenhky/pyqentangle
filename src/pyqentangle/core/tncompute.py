
import numpy as np
import numpy.typing as npt
import tensornetwork as tn


# total density matrix
def bipartitepurestate_densitymatrix(
        bipartitepurestate_tensor: npt.NDArray[np.complex128]
) -> npt.NDArray[np.complex128]:
    """Compute the full density matrix of a discrete bipartite pure state.

    Constructs the rank-4 density matrix :math:`\\rho = |\\Psi\\rangle\\langle\\Psi|` via the
    outer product of the ket and bra tensors using TensorNetwork.

    The resulting tensor has indices ordered as ``(i, j, i', j')``, corresponding to
    :math:`\\rho_{ij,i'j'} = \\psi_{ij}\\,\\psi^*_{i'j'}`.

    Args:
        bipartitepurestate_tensor (npt.NDArray[numpy.complex128]): 2-D complex array of
            shape ``(d1, d2)`` representing a normalised bipartite pure state, where
            element ``[i, j]`` is the coefficient of the basis ket :math:`|ij\\rangle`.

    Returns:
        npt.NDArray[numpy.complex128]: Rank-4 complex array of shape ``(d1, d2, d1, d2)``
        representing the full density matrix :math:`\\rho`.
    """
    ketnode = tn.Node(bipartitepurestate_tensor)
    branode = tn.Node(np.conj(bipartitepurestate_tensor))
    denmat_node = tn.outer_product(ketnode, branode)
    return denmat_node.tensor


def bipartitepurestate_reduceddensitymatrix(
        bipartitepurestate_tensor: npt.NDArray[np.complex128],
        kept: int
) -> npt.NDArray[np.complex128]:
    """Compute the reduced density matrix of one subsystem of a discrete bipartite pure state.

    Traces out the complementary subsystem by contracting the shared index between the ket
    and bra tensors using TensorNetwork, yielding:

    .. math::

        \\rho_A = \\mathrm{Tr}_B(|\\Psi\\rangle\\langle\\Psi|)
        \\quad \\text{or} \\quad
        \\rho_B = \\mathrm{Tr}_A(|\\Psi\\rangle\\langle\\Psi|)

    Args:
        bipartitepurestate_tensor (npt.NDArray[numpy.complex128]): 2-D complex array of
            shape ``(d1, d2)`` representing a normalised bipartite pure state, where
            element ``[i, j]`` is the coefficient of the basis ket :math:`|ij\\rangle`.
        kept (int): Index of the subsystem whose reduced density matrix is returned.
            ``0`` retains the first subsystem (traces out the second);
            ``1`` retains the second subsystem (traces out the first).

    Returns:
        npt.NDArray[numpy.complex128]: 2-D complex array of shape ``(dk, dk)`` representing
        the reduced density matrix of the kept subsystem, where ``dk`` is the dimension of
        that subsystem.

    Raises:
        ValueError: If ``kept`` is not ``0`` or ``1``.
    """
    if not (kept in [0, 1]):
        raise ValueError('kept can only be 0 or 1!')

    ketnode = tn.Node(bipartitepurestate_tensor)
    branode = tn.Node(np.conj(bipartitepurestate_tensor))

    _ = ketnode[1-kept] ^ branode[1-kept]   # defining the edge
    reddenmat_node = ketnode @ branode     # contract

    return reddenmat_node.tensor


def bipartitepurestate_partialtranspose_densitymatrix(
        bipartite_tensor: npt.NDArray[np.complex128],
        pt_subsys: int
) -> npt.NDArray[np.complex128]:
    """Compute the partial transpose of the density matrix of a discrete bipartite pure state.

    Constructs the full rank-4 density matrix :math:`\\rho = |\\Psi\\rangle\\langle\\Psi|` and
    then transposes the indices belonging to the chosen subsystem, implementing the
    partial-transpose map used in the Peres–Horodecki separability criterion.

    For ``pt_subsys=0`` the operation swaps the ket and bra indices of the first subsystem:

    .. math::

        (\\rho^{T_A})_{ij,i'j'} = \\rho_{i'j,ij'}

    For ``pt_subsys=1`` it swaps those of the second subsystem:

    .. math::

        (\\rho^{T_B})_{ij,i'j'} = \\rho_{ij',i'j}

    Args:
        bipartite_tensor (npt.NDArray[numpy.complex128]): 2-D complex array of shape
            ``(d1, d2)`` representing a normalised bipartite pure state, where element
            ``[i, j]`` is the coefficient of the basis ket :math:`|ij\\rangle`.
        pt_subsys (int): Subsystem on which the transpose is applied.
            ``0`` transposes the first subsystem; ``1`` transposes the second.

    Returns:
        npt.NDArray[numpy.complex128]: Rank-4 complex array of shape ``(d1, d2, d1, d2)``
        representing the partially transposed density matrix.

    Raises:
        ValueError: If ``pt_subsys`` is not ``0`` or ``1``.
    """
    if not (pt_subsys in [0, 1]):
        raise ValueError('pt_subsys can only be 0 or 1!')

    ketnode = tn.Node(bipartite_tensor)
    branode = tn.Node(np.conj(bipartite_tensor))
    final_node = tn.outer_product(ketnode, branode)

    e0, e1, e2, e3 = final_node[0], final_node[1], final_node[2], final_node[3]
    if pt_subsys == 0:
        final_node.reorder_edges([e2, e1, e0, e3])
    else:
        final_node.reorder_edges([e0, e3, e2, e1])

    return final_node.tensor


def flatten_bipartite_densitymatrix(
        bipartite_tensor: npt.NDArray[np.complex128]
) -> npt.NDArray[np.complex128]:
    """Flatten a rank-4 bipartite density matrix to a standard rank-2 matrix.

    Merges the two ket indices ``(i, j)`` into a single composite index and the two bra
    indices ``(i', j')`` into another, converting the rank-4 tensor of shape
    ``(d1, d2, d1, d2)`` into a square matrix of shape ``(d1*d2, d1*d2)``.

    This is the conventional reshaping needed before computing eigenvalues, traces, or
    other matrix operations on the density matrix.

    Args:
        bipartite_tensor (npt.NDArray[numpy.complex128]): Rank-4 complex array of shape
            ``(d1, d2, d1, d2)`` representing the density matrix of a bipartite system,
            with index ordering ``(i, j, i', j')``.

    Returns:
        npt.NDArray[numpy.complex128]: 2-D complex array of shape ``(d1*d2, d1*d2)`` –
        the flattened density matrix.
    """
    denmat_node = tn.Node(bipartite_tensor)
    e0, e1, e2, e3 = denmat_node[0], denmat_node[1], denmat_node[2], denmat_node[3]
    tn.flatten_edges([e0, e1])
    tn.flatten_edges([e2, e3])
    return denmat_node.tensor
