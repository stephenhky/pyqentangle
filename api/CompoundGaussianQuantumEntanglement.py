
import numpy as np
import pyqentangle
from functools import partial
import json


def compound_harmonic_fcn(x1, x2, alpha, beta):
    return np.exp(-alpha * (x1 + x2) ** 2) * np.exp(-beta * (x1 - x2) ** 2) * np.sqrt(np.sqrt(8.) / np.pi)


def compute_entanglement(alpha, beta):
    decompositions = pyqentangle.continuous_schmidt_decomposition(partial(compound_harmonic_fcn, alpha=alpha, beta=beta),
                                                                  -10., 10., -10., 10., keep=10)
    von_neumann_entropy = pyqentangle.entanglement_entropy(decompositions)
    participation_ratio = pyqentangle.participation_ratio(decompositions)

    returned_results = {}

    returned_results['schmidt_modes'] = [{'coefficient': mode[0]} for mode in decompositions]
    returned_results['entropy'] = von_neumann_entropy
    returned_results['K'] = participation_ratio

    return returned_results


def run(request):
    a = request['a']
    b = request['b']

    request_json = request.get_json()
    if request.args and 'message' in request.args:
        return request.args.get('message')
    elif request_json and 'message' in request_json:
        return request_json['message']
    else:
        schmidt_modes = pyqentangle.continuous_schmidt_decomposition(partial(compound_harmonic_fcn,
                                                                             alpha=a, beta=b),
                                                                     -10., 10., -10., 10.)
        entropy = pyqentangle.entanglement_entropy(schmidt_modes)
        participation_ratio = pyqentangle.participation_ratio(schmidt_modes)
        returned_results = {'coefficients': [mode[0] for mode in schmidt_modes],
                            'entropy': entropy,
                            'participation_ratio': participation_ratio}

        return json.dumps(returned_results)