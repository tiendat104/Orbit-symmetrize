""" Tests for the EMLP model."""
import torch
from torch.utils.data import DataLoader
from oil.utils.utils import FixedNumpySeed, FixedPytorchSeed
from emlp_pytorch.nn import EMLP
from emlp_pytorch.groups import S, SO, DirectProduct
from emlp_pytorch.reps import vis, sparsify_basis, V, Rep, LazyKron, T
from .equivariance_tests import rel_error, scale_adjusted_rel_error


def equivariance_err(model, mb, repin, repout, group):
    """ Computes the equivariance error of a model on a minibatch mb. """
    x, y = mb
    gs = group.samples(x.size(0))
    rho_gin = torch.vmap(repin(group).rho_dense)(gs)
    rho_gout = torch.vmap(repout(group).rho_dense)(gs)
    y1 = model((rho_gin@x[..., None])[..., 0])
    y2 = (rho_gout@model(x)[..., None])[..., 0]
    return scale_adjusted_rel_error(y1, y2, gs)


def get_dsmb(dsclass, device='cpu'):
    """ Returns a dataset and minibatch for a given dataset class. """
    seed = 2021
    bs = 50
    with FixedNumpySeed(seed), FixedPytorchSeed(seed):
        ds = dsclass(100)
    ds = ds.to(device)
    dataloader = DataLoader(ds, batch_size=min(bs, len(ds)), num_workers=0, pin_memory=False)
    mb = next(iter(dataloader))
    return ds, mb


def test_init_forward_and_equivariance(dsclass, device='cpu'):
    """ Tests that the model can be initialized, forward pass is correct,
        and equivariance is correct. """
    network = EMLP
    ds, mb = get_dsmb(dsclass, device)
    model = network(ds.rep_in, ds.rep_out, group=ds.symmetry).to(device)
    assert equivariance_err(model, mb, ds.rep_in, ds.rep_out, ds.symmetry) < 1e-4, \
        "EMLP failed equivariance test"


def test_utilities(device='cpu'):
    """ Tests that the utilities work. """
    W = V(SO(3).to(device))
    # W = V(DirectProduct(SO(3).to(device), S(6).to(device)))
    vis(W, W)
    Q = (W**2 >> W).equivariant_basis()
    SQ = sparsify_basis(Q)
    A = SQ@(1+torch.arange(SQ.size(-1), device=device)).to(torch.float)
    nunique = len(torch.unique(torch.abs(A)))
    assert nunique in (SQ.size(-1), SQ.size(-1) + 1), "Sparsify failes on SO(3) T3"


def test_bespoke_representations(device='cpu'):
    """ Tests that bespoke representations work. """
    class ProductSubRep(Rep):
        """ A representation of a product group G = G1 x G2 as a sum of two subrepresentations """
        def __init__(self, G, subgroup_id, size):
            """ Produces the representation of the subgroup of G = G1 x G2
                with the index subgroup_id in {0,1} specifying G1 or G2.
                Also requires specifying the size of the representation given by G1.d or G2.d """
            super().__init__()
            self.G = G
            self.index = subgroup_id
            self._size = size
            self.device = device

        def __repr__(self):
            return "V_"+str(self.G).split('x')[self.index]

        def __hash__(self):
            return hash((type(self), (self.G, self.index)))

        def size(self):
            return self._size

        def rho(self, M):
            # Given that M is a LazyKron object, we can just get the argument
            return M.Ms[self.index]

        def drho(self, A):
            return A.Ms[self.index]

        def __call__(self, G):
            # adding this will probably not be necessary in a future release,
            # necessary now because rep is __call__ed in nn.EMLP constructor
            assert self.G == G
            return self
    G1, G2 = SO(3).to(device), S(5).to(device)
    G = G1 * G2

    VSO3 = ProductSubRep(G, 0, G1.d)
    VS5 = ProductSubRep(G, 1, G2.d)
    Vin = VS5 + V(G)
    Vout = VSO3
    str(Vin >> Vout)
    model = EMLP(Vin, Vout, group=G)
    model.to(device)
    input_point = torch.randn(Vin.size(), device=device)*10
    lazy_G_sample = LazyKron([G1.sample(), G2.sample()])

    out1 = model(Vin.rho(lazy_G_sample)@input_point)
    out2 = Vout.rho(lazy_G_sample)@model(input_point)
    assert rel_error(out1, out2) < 1e-4, "EMLP equivariance fails on bespoke productsubrep"
