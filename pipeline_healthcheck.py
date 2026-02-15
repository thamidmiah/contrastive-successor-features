#!/usr/bin/env python3
"""
Pipeline Healthcheck — Contract A (flat buffer)
================================================

Verifies the METRA + SAC + shared-CNN pipeline in two passes:

  PASS 1  Shape checks
  ─────────────────────
  • Env obs is (4,84,84) float32 ∈ [0,1]
  • Replay buffer sample is (B, 28224) flat
  • _encode_obs reshapes flat → (B,4,84,84)
  • CNN output is (B, 512)
  • Q-function input is (B, 512 + dim_option)
  • Policy input  is (B, 512 + dim_option)

  PASS 2  Three backward passes (gradient / detach)
  ──────────────────────────────────────────────────
  (a) Critic   — detach=True  → CNN grads None/0, QF grads ≠ 0
  (b) Actor    — Mode A detach=False → CNN grads ≠ 0, policy grads ≠ 0
                 Mode B detach=True  → CNN grads None/0, policy grads ≠ 0
  (c) TE/METRA — detach=False → CNN grads ≠ 0 always, TE grads ≠ 0

  BONUS  Source-level verification
  ─────────────────────────────────
  • _update_loss_qf calls _encode_obs(…, detach=True)
  • _update_loss_op calls _encode_obs(…, detach=True)  [Mode B]
  • _update_rewards calls _encode_obs(…, detach=False)
  • _optimize_te   steps ['traj_encoder', 'cnn']
  • _optimize_op   steps ['qf'] then ['option_policy']  (never 'cnn')
"""

import sys
import torch
import torch.nn as nn

# ── project imports (dowel_wrapper must precede garage) ──────────────────────
import dowel_wrapper
from garage.torch.modules import MLPModule
from iod.cnn_encoder import NatureCNN

# ── constants (match run_mspacman.sh) ────────────────────────────────────────
B          = 16          # mini-batch size
C, H, W    = 4, 84, 84  # frame-stack × grayscale height × width
FLAT_DIM   = C * H * W  # 28 224
CNN_OUT    = 512
DIM_OPTION = 8           # --dim_option 8 in run_mspacman.sh
ACTION_DIM = 9           # MsPacman discrete actions
HIDDEN     = [1024, 1024]
DEVICE     = 'cpu'

# ── bookkeeping ──────────────────────────────────────────────────────────────
_pass = _fail = 0

def check(cond: bool, tag: str, detail: str = ""):
    global _pass, _fail
    if cond:
        _pass += 1
        print(f"  ✅ {tag}")
    else:
        _fail += 1
        print(f"  ❌ {tag}  —  {detail}")

# ── helpers ──────────────────────────────────────────────────────────────────
def make_cnn():
    return NatureCNN(in_channels=C, output_dim=CNN_OUT).to(DEVICE)

def make_qf():
    return MLPModule(
        input_dim=CNN_OUT + DIM_OPTION,
        output_dim=ACTION_DIM,
        hidden_sizes=HIDDEN,
        hidden_nonlinearity=torch.relu,
    ).to(DEVICE)

def make_te():
    return MLPModule(
        input_dim=CNN_OUT,
        output_dim=DIM_OPTION,
        hidden_sizes=HIDDEN,
        hidden_nonlinearity=torch.relu,
    ).to(DEVICE)

def make_policy():
    return MLPModule(
        input_dim=CNN_OUT + DIM_OPTION,
        output_dim=ACTION_DIM,
        hidden_sizes=HIDDEN,
        hidden_nonlinearity=torch.relu,
    ).to(DEVICE)

def encode_obs(cnn, obs, detach=False):
    """Mirror of METRA._encode_obs — single source of truth."""
    if obs.ndim == 2:
        obs = obs.view(obs.shape[0], C, H, W)
    assert obs.ndim == 4, f"expected 4D, got {obs.shape}"
    out = cnn(obs)
    return out.detach() if detach else out

def zero_all(*modules):
    for m in modules:
        for p in m.parameters():
            p.grad = None

def has_grad(module: nn.Module) -> bool:
    """True if any parameter has a non-None, non-zero gradient."""
    return any(
        p.grad is not None and p.grad.abs().max().item() > 0
        for p in module.parameters()
    )

def has_any_grad(module: nn.Module) -> bool:
    """True if any parameter has a non-None gradient (even all-zero)."""
    return any(p.grad is not None for p in module.parameters())

def grad_info(module: nn.Module, name: str) -> str:
    n_total = sum(1 for _ in module.parameters())
    n_grad  = sum(1 for p in module.parameters() if p.grad is not None)
    mx = max(
        (p.grad.abs().max().item() for p in module.parameters() if p.grad is not None),
        default=0.0,
    )
    return f"{name}: {n_grad}/{n_total} params with grad, max|g|={mx:.6f}"


# ═════════════════════════════════════════════════════════════════════════════
#  PASS 1 — SHAPE CHECKS
# ═════════════════════════════════════════════════════════════════════════════

def test_shapes():
    print("\n" + "═" * 60)
    print("  PASS 1: SHAPE CHECKS")
    print("═" * 60)

    cnn = make_cnn()

    # 1. Env observation
    obs_4d = torch.rand(B, C, H, W, device=DEVICE)
    check(obs_4d.shape == (B, C, H, W), "env obs shape (B,4,84,84)")
    check(obs_4d.dtype == torch.float32, "env obs dtype float32", f"got {obs_4d.dtype}")
    check(0.0 <= obs_4d.min().item() and obs_4d.max().item() <= 1.0,
          "env obs range [0, 1]",
          f"min={obs_4d.min().item():.3f}, max={obs_4d.max().item():.3f}")

    # 2. Replay buffer stores flat
    obs_flat = obs_4d.reshape(B, FLAT_DIM)
    check(obs_flat.shape == (B, FLAT_DIM), f"replay sample shape (B, {FLAT_DIM}) flat")

    # 3. _encode_obs reshapes flat → 4D → CNN → (B,512)
    enc = encode_obs(cnn, obs_flat)
    check(enc.shape == (B, CNN_OUT), f"encode_obs(flat) → (B, {CNN_OUT})")

    # 4. Same result from 4D directly
    enc_4d = encode_obs(cnn, obs_4d)
    diff = (enc - enc_4d).abs().max().item()
    check(diff < 1e-5, "encode_obs(flat) ≡ encode_obs(4D)", f"max Δ = {diff}")

    # 5. CNN rejects raw flat input
    try:
        cnn(obs_flat)
        check(False, "CNN rejects 2D flat input", "no error raised")
    except (AssertionError, RuntimeError):
        check(True, "CNN rejects 2D flat input")

    # 6. Q-function / policy input dims
    cat = torch.cat([enc, torch.zeros(B, DIM_OPTION, device=DEVICE)], dim=-1)
    check(cat.shape == (B, CNN_OUT + DIM_OPTION),
          f"Q / policy input shape (B, {CNN_OUT + DIM_OPTION})")

    qf = make_qf()
    check(qf(cat).shape == (B, ACTION_DIM),
          f"Q-function output (B, {ACTION_DIM})")

    te = make_te()
    check(te(enc).shape == (B, DIM_OPTION),
          f"traj_encoder output (B, {DIM_OPTION})")


# ═════════════════════════════════════════════════════════════════════════════
#  PASS 2 — THREE BACKWARD PASSES
# ═════════════════════════════════════════════════════════════════════════════

def test_grad_critic():
    """(a) Critic backward — detach=True ⇒ CNN grads = None, QF grads ≠ 0."""
    print("\n" + "─" * 60)
    print("  2a  CRITIC  (detach=True)")
    print("─" * 60)

    cnn, qf = make_cnn(), make_qf()
    obs_flat = torch.rand(B, FLAT_DIM, device=DEVICE)
    zero_all(cnn, qf)

    enc = encode_obs(cnn, obs_flat, detach=True)
    cat = torch.cat([enc, torch.zeros(B, DIM_OPTION, device=DEVICE)], dim=-1)
    q = qf(cat)
    loss = q.mean()
    loss.backward()

    print(f"    {grad_info(cnn, 'CNN')}")
    print(f"    {grad_info(qf,  'QF')}")

    check(not has_any_grad(cnn), "CNN grads = None  (detach blocks gradient)")
    check(has_grad(qf),          "QF  grads ≠ 0")


def test_grad_actor_A():
    """(b-A) Actor backward — detach=False ⇒ CNN gets grads (Mode A: actor attached)."""
    print("\n" + "─" * 60)
    print("  2b-A  ACTOR  Mode A (detach=False → CNN trained by actor + TE)")
    print("─" * 60)

    cnn, policy = make_cnn(), make_policy()
    obs_flat = torch.rand(B, FLAT_DIM, device=DEVICE)
    zero_all(cnn, policy)

    enc = encode_obs(cnn, obs_flat, detach=False)
    cat = torch.cat([enc, torch.zeros(B, DIM_OPTION, device=DEVICE)], dim=-1)
    logits = policy(cat)
    loss = logits.mean()
    loss.backward()

    print(f"    {grad_info(cnn,    'CNN')}")
    print(f"    {grad_info(policy, 'Policy')}")

    check(has_grad(cnn),    "CNN    grads ≠ 0  (actor trains encoder)")
    check(has_grad(policy), "Policy grads ≠ 0")


def test_grad_actor_B():
    """(b-B) Actor backward — detach=True ⇒ CNN gets NO grads (Mode B: actor detached)."""
    print("\n" + "─" * 60)
    print("  2b-B  ACTOR  Mode B (detach=True → CNN NOT trained by actor)")
    print("─" * 60)

    cnn, policy = make_cnn(), make_policy()
    obs_flat = torch.rand(B, FLAT_DIM, device=DEVICE)
    zero_all(cnn, policy)

    enc = encode_obs(cnn, obs_flat, detach=True)
    cat = torch.cat([enc, torch.zeros(B, DIM_OPTION, device=DEVICE)], dim=-1)
    logits = policy(cat)
    loss = logits.mean()
    loss.backward()

    print(f"    {grad_info(cnn,    'CNN')}")
    print(f"    {grad_info(policy, 'Policy')}")

    check(not has_any_grad(cnn), "CNN    grads = None  (detach blocks gradient)")
    check(has_grad(policy),      "Policy grads ≠ 0")


def test_grad_metra():
    """(c) TE/METRA backward — detach=False ⇒ CNN always gets grads."""
    print("\n" + "─" * 60)
    print("  2c  METRA / TRAJ ENCODER  (detach=False → CNN always trained)")
    print("─" * 60)

    cnn, te = make_cnn(), make_te()
    obs_flat      = torch.rand(B, FLAT_DIM, device=DEVICE)
    next_obs_flat = torch.rand(B, FLAT_DIM, device=DEVICE)
    zero_all(cnn, te)

    enc      = encode_obs(cnn, obs_flat,      detach=False)
    next_enc = encode_obs(cnn, next_obs_flat, detach=False)

    phi      = te(enc)
    next_phi = te(next_enc)
    phi      = phi      / (phi.norm(dim=-1, keepdim=True) + 1e-8)
    next_phi = next_phi / (next_phi.norm(dim=-1, keepdim=True) + 1e-8)

    delta = next_phi - phi
    option = torch.zeros(B, DIM_OPTION, device=DEVICE)
    option[:, 0] = 1.0
    loss = -(delta * option).sum(dim=-1).mean()
    loss.backward()

    print(f"    {grad_info(cnn, 'CNN')}")
    print(f"    {grad_info(te,  'TE')}")

    check(has_grad(cnn), "CNN grads ≠ 0  (METRA always trains encoder)")
    check(has_grad(te),  "TE  grads ≠ 0")


# ═════════════════════════════════════════════════════════════════════════════
#  BONUS — SOURCE-LEVEL VERIFICATION
# ═════════════════════════════════════════════════════════════════════════════

def test_source_contracts():
    """Verify detach flags and optimizer keys directly from source code."""
    print("\n" + "═" * 60)
    print("  BONUS: SOURCE-LEVEL CONTRACTS")
    print("═" * 60)

    import inspect, re
    from iod.metra import METRA

    src_qf  = inspect.getsource(METRA._update_loss_qf)
    src_op  = inspect.getsource(METRA._update_loss_op)
    src_rew = inspect.getsource(METRA._update_rewards)
    src_ote = inspect.getsource(METRA._optimize_te)
    src_oop = inspect.getsource(METRA._optimize_op)
    src_cnn = inspect.getsource(NatureCNN.forward)

    # Detach flags
    check("_encode_obs(mini_batch['obs'], detach=True)" in src_qf
          or "self._encode_obs(mini_batch['obs'], detach=True)" in src_qf,
          "_update_loss_qf → obs detach=True")

    check("_encode_obs(mini_batch['next_obs'], detach=True)" in src_qf
          or "self._encode_obs(mini_batch['next_obs'], detach=True)" in src_qf,
          "_update_loss_qf → next_obs detach=True")

    check("_encode_obs(mini_batch['obs'], detach=True)" in src_op
          or "self._encode_obs(mini_batch['obs'], detach=True)" in src_op,
          "_update_loss_op → obs detach=True  [Mode B]")

    check("_encode_obs(obs, detach=False)" in src_rew
          or "self._encode_obs(obs, detach=False)" in src_rew,
          "_update_rewards → obs detach=False")

    check("_encode_obs(next_obs, detach=False)" in src_rew
          or "self._encode_obs(next_obs, detach=False)" in src_rew,
          "_update_rewards → next_obs detach=False")

    # Optimizer routing
    check("'traj_encoder', 'cnn'" in src_ote or "'traj_encoder','cnn'" in src_ote,
          "_optimize_te steps ['traj_encoder', 'cnn']")

    check("optimizer_keys=['qf']" in src_oop,
          "_optimize_op steps ['qf'] for critic")

    check("optimizer_keys=['option_policy']" in src_oop,
          "_optimize_op steps ['option_policy'] for actor")

    gd_calls = re.findall(r'_gradient_descent\([^)]+\)', src_oop)
    cnn_in_op = any("'cnn'" in c or '"cnn"' in c for c in gd_calls)
    check(not cnn_in_op,
          "_optimize_op never steps 'cnn'")

    # CNN guard
    check("assert x.ndim == 4" in src_cnn or "assert x.ndim==4" in src_cnn,
          "NatureCNN.forward asserts ndim == 4")


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("╔" + "═" * 58 + "╗")
    print("║  Pipeline Healthcheck — Contract A (flat buffer)        ║")
    print("╚" + "═" * 58 + "╝")

    test_shapes()

    print("\n" + "═" * 60)
    print("  PASS 2: THREE BACKWARD PASSES (gradient / detach)")
    print("═" * 60)
    test_grad_critic()
    test_grad_actor_A()
    test_grad_actor_B()
    test_grad_metra()

    test_source_contracts()

    print("\n" + "═" * 60)
    total = _pass + _fail
    print(f"  {_pass} / {total} PASS    {_fail} FAIL")
    print("═" * 60)

    if _fail:
        print("\n⚠️  Some checks failed — review output above.\n")
        sys.exit(1)
    else:
        print("\n🎉  All checks passed.\n")
        sys.exit(0)
