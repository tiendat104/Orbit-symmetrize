""" Test the EMLP PyTorch implementation. """
import os
import time
from emlp_pytorch.groups import *
from emlp_pytorch.reps import *
from emlp_pytorch.datasets import *
from tests.equivariance_tests import test_sum, test_prod, \
    test_high_rank_representations, \
    test_equivariant_matrix as test_equivariant_matrix_1, \
    test_bilinear_layer, test_large_representations
from tests.product_groups_tests import test_symmetric_mixed_tensor, \
    test_symmetric_mixed_tensor_sum, test_symmetric_mixed_products, \
    test_equivariant_matrix as test_equivariant_matrix_2
from tests.model_tests import test_init_forward_and_equivariance, \
    test_utilities, test_bespoke_representations


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1,5,6,7"
device = 'cuda'


def test_equivariance():
    """ Test the EMLP PyTorch implementation. """
    test_groups = [SO(n) for n in [2, 3, 4]]+[O(n) for n in [2, 3, 4]] + \
        [SU(n) for n in [2, 3, 4]] + [U(n) for n in [2, 3, 4]] + \
        [SL(n) for n in [2, 3, 4]] + [GL(n) for n in [2, 3, 4]] + \
        [C(k) for k in [2, 3, 4, 8]] + [D(k) for k in [2, 3, 4, 8]] + \
        [S(n) for n in [2, 4, 6]] + [Z(n) for n in [2, 4, 6]] + \
        [SO11p(), SO13p(), SO13(), O13()] + [Sp(n) for n in [1, 3]] + \
        [RubiksCube(), Cube(), ZksZnxZn(2, 2), ZksZnxZn(4, 4)]

    test_equivariant_matrix_groups = [
        (SO(3), T(1)+2*T(0), T(1)+T(2)+2*T(0)+T(1)),
        (SO(3), 5*T(0)+5*T(1), 3*T(0)+T(2)+2*T(1)),
        (SO(3), 5*(T(0)+T(1)), 2*(T(0)+T(1))+T(2)+T(1)),
        (SO(4), T(1)+2*T(2), (T(0)+T(3))*T(0)),
        (SO13p(), T(2)+4*T(1, 0)+T(0, 1), 10*T(0) +
         3*T(1, 0)+3*T(0, 1)+T(0, 2)+T(2, 0)+T(1, 1)),
        (Sp(2), (V+2*V**2)*(V.t()+1*V).t() + V.t(), 3*V**0 + V + V*V.t()),
        (SU(3), T(2, 0)+T(1, 1)+T(0)+2*T(0, 1), T(1, 1)+V+V.t()+T(0)+T(2, 0)+T(0, 2))]

    test_bilinear_layer_groups = [
        (SO(3), 5*T(0)+5*T(1), 3*T(0)+T(2)+2*T(1)),
        (SO13p(), 4*T(1, 0), 10*T(0)+3*T(1, 0)+3*T(0, 1)+T(0, 2)+T(2, 0)+T(1, 1))]

    for group in test_groups:
        print(f"Testing sum for group {group}")
        try:
            test_sum(group, device)
        except AssertionError as e:
            print(e)

    for group in [group for group in test_groups if group.d < 5]:
        print(f"Testing prod for group {group}")
        try:
            test_prod(group, device)
        except AssertionError as e:
            print(e)

    for group in test_groups:
        print(f"Testing representations for group {group}")
        try:
            test_high_rank_representations(group, device)
        except AssertionError as e:
            print(e)

    for group, repin, repout in test_equivariant_matrix_groups:
        print(f"Testing equivariant matrix for group {group}, repin {repin}, repout {repout}")
        try:
            test_equivariant_matrix_1(group, repin, repout, device)
        except AssertionError as e:
            print(e)

    for group, repin, repout in test_bilinear_layer_groups:
        print(f"Testing bilinear layer for group {group}, repin {repin}, repout {repout}")
        try:
            test_bilinear_layer(group, repin, repout, device)
        except AssertionError as e:
            print(e)

    for group in test_groups:
        print(f"Testing large representations for group {group}")
        try:
            test_large_representations(group, device)
        except AssertionError as e:
            print(e)


def test_product_groups():
    """ Test the EMLP PyTorch implementation. """
    test_groups = [(SO(3), S(5)), (S(5), SO(3))]

    for group1, group2 in test_groups:
        print(f"Testing symmetric mixed tensor for groups {(group1, group2)}")
        try:
            test_symmetric_mixed_tensor(group1, group2, device)
        except AssertionError as e:
            print(e)

    for group1, group2 in test_groups:
        print(f"Testing symmetric mixed tensor sum for groups {(group1, group2)}")
        try:
            test_symmetric_mixed_tensor_sum(group1, group2, device)
        except AssertionError as e:
            print(e)

    for group1, group2 in test_groups:
        print(f"Testing symmetric mixed products for groups {(group1, group2)}")
        try:
            test_symmetric_mixed_products(group1, group2, device)
        except AssertionError as e:
            print(e)

    for group1, group2 in test_groups:
        print(f"Testing equivariance matrix for groups {(group1, group2)}")
        try:
            test_equivariant_matrix_2(group1, group2, device)
        except AssertionError as e:
            print(e)


def test_model():
    """ Test the EMLP PyTorch implementation. """
    test_dsclasses = [Inertia, O5Synthetic, ParticleInteraction, InvertedCube]

    for dsclass in test_dsclasses:
        print(f"Testing initialization, forward, and equivariance for dataset {(dsclass)}")
        try:
            test_init_forward_and_equivariance(dsclass, device)
        except AssertionError as e:
            print(e)

    print("Testing utilities")
    try:
        test_utilities(device)
    except AssertionError as e:
        print(e)

    print("Testing beskope representations")
    try:
        test_bespoke_representations(device)
    except AssertionError as e:
        print(e)


if __name__ == "__main__":
    tic = time.time()
    test_equivariance()
    test_product_groups()
    test_model()
    print(f"Total time: {time.time() - tic:.2f} seconds")
