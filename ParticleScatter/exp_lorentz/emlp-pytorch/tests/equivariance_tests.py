""" Tests for equivariance of representations and neural networks. """
import logging
import torch
from emlp_pytorch.groups import *
from emlp_pytorch.reps import *
from emlp_pytorch.nn import uniform_rep


def rel_error(t1, t2):
    """ Computes the relative error of two tensors. """
    error = torch.sqrt(torch.mean(torch.abs(t1-t2)**2))
    scale = torch.sqrt(torch.mean(torch.abs(t1)**2)) + \
        torch.sqrt(torch.mean(torch.abs(t2)**2))
    return error/torch.clamp(scale, min=1e-7)


def scale_adjusted_rel_error(t1, t2, g):
    """ Computes the relative error of two tensors t1 and t2 under the action of g. """
    error = torch.sqrt(torch.mean(torch.abs(t1-t2)**2))
    tscale = torch.sqrt(torch.mean(torch.abs(t1)**2)) + \
        torch.sqrt(torch.mean(torch.abs(t2)**2))
    gscale = torch.sqrt(torch.mean(torch.abs(g-torch.eye(g.size(-1), device=t1.device))**2))
    scale = torch.max(tscale, gscale)
    return error/torch.clamp(scale, min=1e-7)


def equivariance_error(W, repin, repout, G):
    """ Computes the equivariance relative error rel_err(Wρ₁(g),ρ₂(g)W)
        of the matrix W (dim(repout),dim(repin))
        according to the input and output representations and group G. """
    N = 5
    gs = G.samples(N)
    ring = torch.vmap(repin.rho_dense)(gs)
    routg = torch.vmap(repout.rho_dense)(gs)
    equiv_err = scale_adjusted_rel_error(W@ring, routg@W, gs)
    return equiv_err


def test_sum(G, device='cpu'):
    """ Tests equivariance of the sum of representations. """
    G = G.to(device)
    N = 5
    rep = T(0, 2)+3*(T(0, 0)+T(1, 0))+T(0, 0)+T(1, 1)+2*T(1, 0)+T(0, 2)+T(0, 1)+3*T(0, 2)+T(2, 0)
    rep = rep(G)
    if G.num_constraints()*rep.size() > 1e11 or rep.size()**2 > 10**7:
        return
    P = rep.equivariant_projector()
    v = torch.rand(rep.size(), device=device)
    v = P@v
    gs = G.samples(N)
    gv = (torch.vmap(rep.rho_dense)(gs)*v).sum(-1)
    err = torch.vmap(scale_adjusted_rel_error)(gv, v+torch.zeros_like(gv), gs).mean()
    assert err < 1e-4, f"Symmetric vector fails err {err:.3e} with G={G}"


def test_prod(G, device='cpu'):
    """ Tests equivariance of the product of representations. """
    G = G.to(device)
    N = 5
    rep = T(0, 1)*T(0, 0)*T(1, 0)**2*T(1, 0)*T(0, 0)**3*T(0, 1)
    rep = rep(G)
    if G.num_constraints()*rep.size() > 1e11 or rep.size()**2 > 10**7:
        return
    Q = rep.equivariant_basis()
    v = Q@torch.rand(Q.size(-1), device=device)
    gs = G.samples(N)
    gv = (torch.vmap(rep.rho_dense)(gs)*v).sum(-1)
    err = torch.vmap(scale_adjusted_rel_error)(gv, v+torch.zeros_like(gv), gs).mean()
    assert err < 1e-4, f"Symmetric vector fails err {err:.3e} with G={G}"


def test_high_rank_representations(G, device='cpu'):
    """ Tests equivariance of the sum of representations. """
    G = G.to(device)
    N = 5
    r = 10
    for p in range(r+1):
        for q in range(r-p+1):
            if G.num_constraints()*G.d**(3*(p+q)) > 1e11:
                continue
            if G.is_orthogonal and q > 0:
                continue
            rep = T(p, q)(G)
            P = rep.equivariant_projector()
            v = torch.rand(rep.size(), device=device)
            v = P@v
            g = torch.vmap(rep.rho_dense)(G.samples(N))
            gv = (g*v).sum(-1)
            err = torch.vmap(scale_adjusted_rel_error)(
                gv, v+torch.zeros_like(gv), g).mean()
            if torch.isnan(err):
                continue  # deal with nans on cpu later
            assert err < 1e-4, f"Symmetric vector fails err {err:.3e} with T{p,q} and G={G}"
            logging.info("Success with T%r and G=%r", (p,q), G)


def test_equivariant_matrix(G, repin, repout, device='cpu'):
    """ Tests equivariance """
    G = G.to(device)
    N = 5
    repin = repin(G)
    repout = repout(G)
    # repW = repout*repin.t()
    repW = repin >> repout
    P = repW.equivariant_projector()
    W = torch.rand(repout.size(), repin.size(), device=device)
    W = (P@W.reshape(-1)).reshape(*W.shape)

    x = torch.rand(N, repin.size(), device=device)
    gs = G.samples(N)
    ring = torch.vmap(repin.rho_dense)(gs)
    routg = torch.vmap(repout.rho_dense)(gs)
    ring, x = dtype_cast(ring, x)
    gx = (ring@x[..., None])[..., 0]
    gx, W = dtype_cast(gx, W)
    Wgx = gx@W.t()
    xWT = x@W.t()
    routg, xWT = dtype_cast(routg, xWT)
    gWx = (routg@xWT[..., None])[..., 0]
    equiv_err = rel_error(Wgx, gWx)
    assert equiv_err < 1e-4, f"Equivariant gWx=Wgx fails err {equiv_err:.3e} with G={G}"

    gvecW = (torch.vmap(repW.rho_dense)(gs)*W.reshape(-1)).sum(-1)
    W, gvecW = dtype_cast(W, gvecW)
    gs, gvecW = dtype_cast(gs, gvecW)
    gWerr = torch.vmap(scale_adjusted_rel_error)(
        gvecW, W.reshape(-1)+torch.zeros_like(gvecW), gs).mean()
    assert gWerr < 1e-4, f"Symmetric gvec(W)=vec(W) fails err {gWerr:.3e} with G={G}"


def test_bilinear_layer(G, repin, repout, device='cpu'):
    """ Test equivariance of bilinear layers """
    G = G.to(device)
    N = 5
    repin = repin(G)
    repout = repout(G)
    repW = repout*repin.t()
    Wdim, P = bilinear_weights(repout, repin)
    x = torch.rand(N, repin.size(), device=device)
    gs = G.samples(N)
    ring = torch.vmap(repin.rho_dense)(gs)
    routg = torch.vmap(repout.rho_dense)(gs)
    ring, x = dtype_cast(ring, x)
    gx = (ring@x[..., None])[..., 0]

    W = torch.rand(Wdim, device=device)
    gx, W = dtype_cast(gx, W)
    W_x = P(W, x)
    Wxx = (W_x@x[..., None])[..., 0]
    routg, Wxx = dtype_cast(routg, Wxx)
    gWxx = (routg@Wxx[..., None])[..., 0]
    Wgxgx = (P(W, gx)@gx[..., None])[..., 0]
    equiv_err = rel_error(Wgxgx, gWxx)
    assert equiv_err < 1e-4, f"Bilinear Equivariance fails err {equiv_err:.3e} with G={G}"


def test_large_representations(G, device='cpu'):
    """ Test equivariance of large representations """
    G = G.to(device)
    N = 5
    ch = 256
    rep = repin = repout = uniform_rep(ch, G)
    repW = rep >> rep
    P = repW.equivariant_projector()
    W = torch.rand(repout.size(), repin.size(), device=device)
    W = (P@W.reshape(-1)).reshape(*W.shape)

    x = torch.rand(N, repin.size(), device=device)
    gs = G.samples(N)
    ring = torch.vmap(repin.rho_dense)(gs)
    routg = torch.vmap(repout.rho_dense)(gs)
    ring, x = dtype_cast(ring, x)
    gx = (ring@x[..., None])[..., 0]
    gx, W = dtype_cast(gx, W)
    Wgx = gx@W.t()
    xWT = x@W.t()
    routg, xWT = dtype_cast(routg, xWT)
    gWx = (routg@xWT[..., None])[..., 0]
    equiv_err = torch.vmap(scale_adjusted_rel_error)(Wgx, gWx, gs).mean()
    assert equiv_err < 1e-4, f"Large Rep Equivariant gWx=Wgx fails err {equiv_err:.3e} with G={G}"
    logging.info("Success with G=%r", G)
