""" Datasets for testing equivariant networks """
import torch
import numpy as np

from .reps import Scalar, Vector, T
from .groups import O, Lorentz, RubiksCube, Cube
from .interface import GroupAugmentation


class Inertia():
    """ Inertia tensor dataset """

    def __init__(self, N=1024, k=5):
        self.dim = (1+3)*k
        self.X = np.random.randn(N, self.dim)
        self.X[:, :k] = np.log(1+np.exp(self.X[:, :k]))  # Masses
        mi = self.X[:, :k]
        ri = self.X[:, k:].reshape(-1, k, 3)
        I = np.eye(3)
        r2 = (ri**2).sum(-1)[..., None, None]
        inertia = (mi[:, :, None, None] *
                   (r2*I - ri[..., None]*ri[..., None, :])).sum(1)
        self.Y = inertia.reshape(-1, 9)
        self.rep_in = k*Scalar+k*Vector
        self.rep_out = T(2)
        self.symmetry = O(3)
        # One has to be careful computing offset and scale in a way so that standardizing
        # does not violate equivariance
        Xmean = self.X.mean(0)
        Xmean[k:] = 0
        Xstd = np.zeros_like(Xmean)
        Xstd[:k] = np.abs(self.X[:, :k]).mean(0)  # .std(0)
        Xstd[k:] = (np.abs(self.X[:, k:].reshape(N, k, 3)).mean(
            (0, 2))[:, None] + np.zeros((k, 3))).reshape(k*3)
        Ymean = 0*self.Y.mean(0)
        # Ystd = np.sqrt(((self.Y-Ymean)**2).mean((0,1)))+ np.zeros_like(Ymean)
        Ystd = np.abs(self.Y-Ymean).mean((0, 1)) + np.zeros_like(Ymean)
        self.stats = 0, 1, 0, 1  # Xmean,Xstd,Ymean,Ystd

    def __getitem__(self, i):
        return (torch.tensor(self.X[i]), torch.tensor(self.Y[i]))

    def __len__(self):
        return self.X.shape[0]

    def default_aug(self, model, n_samples=1, test_aug=False, test_n_samples=1):
        """ Group equivariant data augmentation """
        return GroupAugmentation(model, self.rep_in, self.rep_out, self.symmetry,
                                 n_samples, test_aug, test_n_samples)


class O5Synthetic():
    """ Synthetic dataset with O(5) symmetry """

    def __init__(self, N=1024):
        d = 5
        self.dim = 2*d
        self.X = np.random.randn(N, self.dim)
        ri = self.X.reshape(-1, 2, 5)
        r1, r2 = ri.transpose(1, 0, 2)
        self.Y = np.sin(np.sqrt((r1**2).sum(-1)))-.5*np.sqrt((r2**2).sum(-1))**3 + \
            (r1*r2).sum(-1)/(np.sqrt((r1**2).sum(-1))*np.sqrt((r2**2).sum(-1)))
        self.rep_in = 2*Vector
        self.rep_out = Scalar
        self.symmetry = O(d)
        self.Y = self.Y[..., None]
        # One has to be careful computing mean and std in a way so that standardizing
        # does not violate equivariance
        Xmean = self.X.mean(0)  # can add and subtract arbitrary tensors
        Xscale = (np.sqrt((self.X.reshape(N, 2, d)**2).mean((0, 2)))
                  [:, None]+0*ri[0]).reshape(self.dim)
        self.stats = 0, Xscale, self.Y.mean(axis=0), self.Y.std(axis=0)

    def __getitem__(self, i):
        return (self.X[i], self.Y[i])

    def __len__(self):
        return self.X.shape[0]

    def default_aug(self, model):
        """ Group equivariant data augmentation """
        return GroupAugmentation(model, self.rep_in, self.rep_out, self.symmetry)


class ParticleInteraction():
    """ Electron muon e^4 interaction"""

    def __init__(self, N=1024):
        self.dim = 4*4
        self.rep_in = 4*Vector
        self.rep_out = Scalar
        self.X = torch.randn(N, self.dim)/4
        P = self.X.reshape(N, 4, 4)
        p1, p2, p3, p4 = P.permute(1, 0, 2)
        eng = torch.diag(torch.tensor([1., -1., -1., -1.]))

        def dot(v1, v2):
            return ((v1@eng)*v2).sum(-1)
        Le = (p1[:, :, None]*p3[:, None, :] -
              (dot(p1, p3)-dot(p1, p1))[:, None, None]*eng)
        Lmu = ((p2@eng)[:, :, None]*(p4@eng)[:, None, :] -
               (dot(p2, p4)-dot(p2, p2))[:, None, None]*eng)
        M = 4*(Le*Lmu).sum(-1).sum(-1)
        self.Y = M
        self.symmetry = Lorentz()
        self.Y = self.Y[..., None]
        # One has to be careful computing mean and std in a way so that standardizing
        # does not violate equivariance
        self.Xscale = (self.X.reshape(N, 4, 4) @ eng @ self.X.reshape(N, 4, 4).transpose(1, 2)).abs().mean(-1).mean(0).sqrt()
        self.Xscale = (self.Xscale[:, None] + torch.zeros((4, 4))).reshape(-1)
        self.stats = 0, self.Xscale, self.Y.mean(axis=0), self.Y.std(axis=0)

    def __getitem__(self, i):
        return (self.X[i], self.Y[i])

    def __len__(self):
        return self.X.shape[0]

    def default_aug(self, model):
        """ Group equivariant data augmentation """
        return GroupAugmentation(model, self.rep_in, self.rep_out, self.symmetry)


class InvertedCube():
    """ Inverted cube dataset """
    def __init__(self, train=True):
        # TODO: finish implementing this simple dataset
        solved_state = torch.eye(6)
        parity_perm = torch.tensor([5, 3, 4, 1, 2, 0])
        parity_state = solved_state[:, parity_perm]

        labels = torch.tensor([1, 0]).to(torch.int)
        self.X = torch.zeros((2, 6*6))
        self.X[0] = solved_state.reshape(-1)
        self.X[1] = parity_state.reshape(-1)
        self.Y = labels
        self.symmetry = Cube()
        self.rep_in = 6*Vector
        self.rep_out = 2*Scalar
        self.stats = (0, 1)
        if not train:  # Scramble the cubes for test time
            gs = self.symmetry.samples(100)
            self.X = torch.repeat(self.X, 50, axis=0).reshape(100, 6, 6)@gs
            self.Y = torch.repeat(self.Y, 50, axis=0)
            p = torch.randperm(100)
            self.X = self.X[p].reshape((100, -1))
            self.Y = self.Y[p].reshape((100,))
        self.X = self.X.clone().detach()
        self.Y = self.Y.clone().detach()

    def __getitem__(self, i):
        return (self.X[i], self.Y[i])

    def __len__(self):
        return self.X.shape[0]


# Ways of constructing invalid Rubik's cubes
# (see https://ruwix.com/the-rubiks-cube/unsolvable-rubiks-cube-invalid-scramble/)

def UBedge_flip(state):
    """ # invalid edge flip on Rubiks state (6,48)"""
    UB_edge = torch.tensor([1, 3*8+1])  # Top middle of U, top middle of B
    edge_flip = state.copy()
    edge_flip[:, UB_edge] = edge_flip[:, torch.roll(UB_edge, 1)]
    return edge_flip


def ULBcorner_rot(state, i=1):
    """ Invalid rotated corner with ULB corner rotation """
    ULB_corner_ids = torch.tensor([0, 4*8, 3*8+2])  # top left of U, top left of L, top right of B
    rotated_corner_state = state.copy()
    rotated_corner_state[:, ULB_corner_ids] = rotated_corner_state[:, torch.roll(ULB_corner_ids, i)]
    return rotated_corner_state


def LBface_swap(state):
    """ Invalid piece swap between L center top and B center top """
    L_B_center_top_faces = torch.tensor([4*8+1, 3*8+1])
    piece_swap = state.copy()
    piece_swap[:, L_B_center_top_faces] = piece_swap[:, torch.roll(L_B_center_top_faces, 1)]
    return piece_swap


class BrokenRubiksCube():
    """ Binary classification problem of predicting whether a Rubik's cube configuration
        is solvable or 'broken' and not able to be solved by transformations from the group
        e.g. by removing and twisting a corner before replacing.
        Dataset is generated by taking several hand identified simple instances of solvable
        and unsolvable cubes, and then scrambling them.

        Features are represented as 6xT(1) tensors of the Rubiks Group (one hot for each color)"""

    def __init__(self, train=True):
        # start with a valid configuration

        solved_state = torch.zeros((6, 48))
        for i in range(6):
            solved_state[i, 8*i:8*(i+1)] = 1

        # equivalence_classes = np.vstack([t3(t2(t1(solved_state))) for t1,t2,t3 in transforms])
        labels = torch.zeros((22,))
        labels[:11] = 1  # First configuration is solvable
        labels[11:] = 0  # all others are not
        # duplicate solvable example 11 times for convencience (balance)
        self.X = torch.zeros((22, 6*48))
        # equivalence_classes.reshape(12,-1)[:1]
        self.X[:11] = solved_state.reshape(-1)
        parity_perm = torch.tensor([5, 3, 4, 1, 2, 0])
        # equivalence_classes.reshape(12,-1)[1:]
        self.X[11:] = solved_state.reshape(6, 6, 8)[:, parity_perm].reshape(-1)
        self.Y = labels
        self.symmetry = RubiksCube()
        self.rep_in = 6*Vector
        self.rep_out = 2*Scalar
        self.stats = (0, 1)
        if not train:  # Scramble the cubes for test time
            gs = self.symmetry.samples(440)
            self.X = torch.repeat(self.X, 20, axis=0).reshape(440, 6, 48)@gs
            self.Y = torch.repeat(self.Y, 20, axis=0)
            p = torch.randperm(440)
            self.X = self.X[p].reshape((440, -1))
            self.Y = self.Y[p].reshape((440,))
        self.X = self.X.clone().detach()
        self.Y = self.Y.clone().detach()

    def __getitem__(self, i):
        return (self.X[i], self.Y[i])

    def __len__(self):
        return self.X.shape[0]
