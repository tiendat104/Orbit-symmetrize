""" Tests for product groups. """
import torch
from emlp_pytorch.groups import *
from emlp_pytorch.reps import *
from .equivariance_tests import rel_error


def test_symmetric_mixed_tensor(G1, G2, device='cpu'):
    """ Tests equivariance of a symmetric tensor product representation. """
    G1 = G1.to(device)
    G2 = G2.to(device)
    N = 5
    rep = T(2)(G1)*T(1)(G2)
    P = rep.equivariant_projector()
    v = torch.rand(rep.size(), device=device)
    v = P@v
    samples = {G1: G1.samples(N), G2: G2.samples(N)}
    gv = (torch.vmap(rep.rho_dense)(samples)*v).sum(-1)
    err = rel_error(gv, v+torch.zeros_like(gv))
    assert err < 3e-5, f"Symmetric vector fails err {err:.3e} with G={G1}x{G2}"


def test_symmetric_mixed_tensor_sum(G1, G2, device='cpu'):
    """ Tests equivariance of a symmetric tensor product representation. """
    G1 = G1.to(device)
    G2 = G2.to(device)
    N = 5
    rep = T(2)(G1)*T(1)(G2) + 2*T(0)(G1)*T(2)(G2)+T(1)(G1) + T(1)(G2)
    P = rep.equivariant_projector()
    v = torch.rand(rep.size(), device=device)
    v = P@v
    samples = {G1: G1.samples(N), G2: G2.samples(N)}
    gv = (torch.vmap(rep.rho_dense)(samples)*v).sum(-1)
    err = rel_error(gv, v+torch.zeros_like(gv))
    assert err < 3e-5, f"Symmetric vector fails err {err:.3e} with G={G1}x{G2}"


def test_symmetric_mixed_products(G1, G2, device='cpu'):
    """ Tests equivariance of a symmetric tensor product representation. """
    G1 = G1.to(device)
    G2 = G2.to(device)
    N = 5
    rep1 = (T(0)+2*T(1)+T(2))(G1)
    rep2 = (T(0)+T(1))(G2)
    rep = rep2*rep1.t()
    P = rep.equivariant_projector()
    v = torch.rand(rep.size(), device=device)
    v = P@v
    W = v.reshape((rep2.size(), rep1.size()))
    x = torch.rand(N, rep1.size(), device=device)
    g1s = G1.samples(N)
    g2s = G2.samples(N)
    ring = torch.vmap(rep1.rho_dense)(g1s)
    routg = torch.vmap(rep2.rho_dense)(g2s)
    gx = (ring@x[..., None])[..., 0]
    Wgx = gx@W.t()
    gWx = (routg@(x@W.t())[..., None])[..., 0]
    equiv_err = rel_error(Wgx, gWx)
    assert equiv_err < 1e-5, f"Equivariant gWx=Wgx fails err {equiv_err:.3e} with G={G1}x{G2}"
    samples = {G1: g1s, G2: g2s}
    gv = (torch.vmap(rep.rho_dense)(samples)*v).sum(-1)
    err = rel_error(gv, v+torch.zeros_like(gv))
    assert err < 3e-5, f"Symmetric vector fails err {err:.3e} with G={G1}x{G2}"


def test_equivariant_matrix(G1, G2, device='cpu'):
    """ Tests equivariance of a symmetric tensor product representation. """
    G1 = G1.to(device)
    G2 = G2.to(device)
    N = 5
    repin = T(2)(G2) + 3*T(0)(G1) + T(1)(G2)+2*T(2)(G1)*T(1)(G2)
    repout = (T(1)(G1) + T(2)(G1)*T(0)(G2) + T(1)(G1) * T(1)(G2) + T(0)(G1)+T(2)(G1)*T(1)(G2))
    repW = repout*repin.t()
    P = repW.equivariant_projector()
    W = torch.rand(repout.size(), repin.size(), device=device)
    W = (P@W.reshape(-1)).reshape(*W.shape)

    x = torch.rand(N, repin.size(), device=device)
    samples = {G1: G1.samples(N), G2: G2.samples(N)}
    ring = torch.vmap(repin.rho_dense)(samples)
    routg = torch.vmap(repout.rho_dense)(samples)
    gx = (ring@x[..., None])[..., 0]
    Wgx = gx@W.t()
    gWx = (routg@(x@W.t())[..., None])[..., 0]
    equiv_err = rel_error(Wgx, gWx)
    assert equiv_err < 3e-5, f"Equivariant gWx=Wgx fails err {equiv_err:.3e} with G={G1}x{G2}"
