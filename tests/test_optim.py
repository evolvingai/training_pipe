"""MuSGD: Newton-Schulz orthogonalization, SGD-half parity, and state round-trip."""

import pytest
import torch

from trainer.detection.engine.musgd import MuSGD, newton_schulz


def _quadratic_step(params, curvature):
    """Gradient of 0.5 * sum(curvature * p^2), written into p.grad."""
    for p in params:
        p.grad = curvature * p.detach()


@pytest.mark.parametrize("shape", [(64, 32), (32, 64), (48, 48)])
def test_newton_schulz_preserves_shape(shape):
    g = torch.randn(*shape, generator=torch.Generator().manual_seed(0))
    out = newton_schulz(g)
    assert out.shape == g.shape and torch.isfinite(out).all()


@pytest.mark.parametrize("shape", [(64, 32), (32, 64)])
def test_newton_schulz_equalizes_singular_values(shape):
    """Non-square Gaussians have a spectrum bounded away from zero (Marchenko-Pastur),
    so 5 steps reliably pull every singular value into a narrow band around 1. Square
    matrices are excluded: their smallest singular value concentrates near 0, where
    five steps of a quintic are not always enough to lift it."""
    g = torch.randn(*shape, generator=torch.Generator().manual_seed(0))
    sv = torch.linalg.svdvals(newton_schulz(g))
    assert sv.min() > 0.3 and sv.max() < 1.7
    # The input is far from orthogonal -- confirms the assertion would catch a no-op.
    assert torch.linalg.svdvals(g / g.norm()).min() < 0.3


def test_newton_schulz_rejects_non_matrix():
    with pytest.raises(ValueError):
        newton_schulz(torch.randn(16, 8, 3, 3))


def test_newton_schulz_zero_input_is_zero_not_nan():
    out = newton_schulz(torch.zeros(8, 16))
    assert torch.isfinite(out).all() and out.abs().max() == 0.0


def test_conv_weight_orthogonalizes_through_flatten():
    w = torch.randn(16, 8, 3, 3, generator=torch.Generator().manual_seed(1))
    out = newton_schulz(w.reshape(w.shape[0], -1)).reshape(w.shape)
    assert out.shape == w.shape and torch.isfinite(out).all()


@pytest.mark.parametrize("nesterov", [False, True])
@pytest.mark.parametrize("weight_decay", [0.0, 0.01])
def test_sgd_half_matches_torch_sgd_when_muon_disabled(nesterov, weight_decay):
    """use_muon=False must reduce MuSGD exactly to torch.optim.SGD."""
    torch.manual_seed(3)
    init = torch.randn(10)
    curvature = torch.rand(10) + 0.5
    a, b = init.clone().requires_grad_(), init.clone().requires_grad_()
    kwargs = dict(lr=0.05, momentum=0.9, nesterov=nesterov, weight_decay=weight_decay)
    ours = MuSGD([a], use_muon=False, **kwargs)
    theirs = torch.optim.SGD([b], **kwargs)
    for _ in range(20):
        _quadratic_step([a], curvature)
        _quadratic_step([b], curvature)
        ours.step()
        theirs.step()
    assert torch.allclose(a, b, atol=1e-6)


def test_muon_weight_zero_reduces_to_sgd_scaled_by_sgd_weight():
    torch.manual_seed(4)
    init = torch.randn(8, 4)
    curvature = torch.rand(8, 4) + 0.5
    a, b = init.clone().requires_grad_(), init.clone().requires_grad_()
    ours = MuSGD([a], lr=0.1, momentum=0.9, nesterov=True, use_muon=True,
                 muon_weight=0.0, sgd_weight=0.5)
    theirs = torch.optim.SGD([b], lr=0.05, momentum=0.9, nesterov=True)
    for _ in range(15):
        _quadratic_step([a], curvature)
        _quadratic_step([b], curvature)
        ours.step()
        theirs.step()
    assert torch.allclose(a, b, atol=1e-6)


def test_muon_group_converges_on_a_convex_problem():
    torch.manual_seed(5)
    p = torch.randn(12, 6).requires_grad_()
    curvature = torch.rand(12, 6) + 0.5
    opt = MuSGD([p], lr=0.02, momentum=0.9, nesterov=True, use_muon=True,
                muon_weight=0.2, sgd_weight=1.0)
    start = float((curvature * p.detach() ** 2).sum())
    for _ in range(50):
        _quadratic_step([p], curvature)
        opt.step()
    assert float((curvature * p.detach() ** 2).sum()) < start * 0.5


def test_muon_allocates_both_buffers_only_where_needed():
    matrix = torch.randn(6, 4).requires_grad_()
    vector = torch.randn(6).requires_grad_()
    opt = MuSGD([{"params": [matrix], "use_muon": True}, {"params": [vector]}],
                lr=0.01, momentum=0.9)
    _quadratic_step([matrix, vector], torch.tensor(1.0))
    opt.step()
    assert set(opt.state[matrix]) == {"muon_buf", "sgd_buf"}
    assert set(opt.state[vector]) == {"sgd_buf"}


def test_state_dict_round_trip_resumes_the_same_trajectory():
    torch.manual_seed(6)
    init = torch.randn(8, 4)
    curvature = torch.rand(8, 4) + 0.5
    kwargs = dict(lr=0.05, momentum=0.9, nesterov=True, use_muon=True,
                  muon_weight=0.3, sgd_weight=0.8, weight_decay=0.001)

    straight = init.clone().requires_grad_()
    opt_a = MuSGD([straight], **kwargs)
    for _ in range(10):
        _quadratic_step([straight], curvature)
        opt_a.step()

    interrupted = init.clone().requires_grad_()
    opt_b = MuSGD([interrupted], **kwargs)
    for _ in range(5):
        _quadratic_step([interrupted], curvature)
        opt_b.step()
    saved = opt_b.state_dict()
    # Blend weights must survive the round-trip: they live in the param groups, not
    # on the instance, precisely so a resumed run restores the blend it trained with.
    assert saved["param_groups"][0]["muon_weight"] == 0.3
    assert saved["param_groups"][0]["sgd_weight"] == 0.8

    resumed = MuSGD([interrupted], lr=1.0, momentum=0.0, use_muon=False)
    resumed.load_state_dict(saved)
    for _ in range(5):
        _quadratic_step([interrupted], curvature)
        resumed.step()

    assert torch.allclose(straight, interrupted, atol=1e-6)


def test_rejects_invalid_hyperparameters():
    p = [torch.zeros(4, 4).requires_grad_()]
    for kwargs in ({"lr": -1.0}, {"momentum": 1.0}, {"weight_decay": -0.1},
                   {"muon_weight": -0.1}, {"sgd_weight": -1.0}):
        with pytest.raises(ValueError):
            MuSGD(p, **kwargs)
