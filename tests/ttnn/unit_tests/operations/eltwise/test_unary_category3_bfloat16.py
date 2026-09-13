# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import torch.nn.functional as F
import ttnn
from tests.ttnn.utils_for_testing import assert_with_ulp, assert_with_pcc, assert_allclose
from tests.ttnn.unit_tests.operations.eltwise.eltwise_test_utils import (
    generate_bfloat16_bits,
    generate_bfloat16_bits_in_range,
    to_tt_tensor,
    SMALLEST_NORMAL_BF16,
)

pytestmark = pytest.mark.use_module_device

MAX_BF16 = float(torch.finfo(torch.bfloat16).max)

"""
Category 3: ops with a fast_and_approximate_mode parameter.
sqrt, rsqrt, exp, erf, gelu, log, log10, log2, log1p, mish.

Each op is swept in both modes over exhaustive normal-bfloat16 input (same
65,536-bit-pattern helpers as category1/2/5):
  · fast_and_approximate_mode = False  (accurate / default SFPU path)
  · fast_and_approximate_mode = True   (fast approximate path)

Accuracy criteria (grounded in the already-merged, hardware-verified tests
these consolidate):
  sqrt/rsqrt : ULP <= 1/2 accurate, ULP <= 2 fast over [1,100]; x<0 (x<=0 for
               rsqrt) -> non-finite                    [test_math, test_unary]
  exp        : ULP <= 1 accurate, PCC >= 0.999 fast, over [-87, 88.5]; exact
               0/+inf tails checked separately          [test_unary_category1]
  erf        : ULP <= 2 both modes, over [-10, 10]         [test_math, test_unary_ops_ttnn]
  gelu       : ULP <= 10 accurate (all normals, FTZ band excluded), PCC >= 0.999
               fast over [-10, 10]                       [test_activation, test_unary_ops_ttnn]
  log/log2/log10/log1p : ULP <= 1/1/2/1 accurate, allclose(atol=0.0625) fast
               over [1,100]; x out-of-domain -> non-finite [test_math, test_unary(_ops_ttnn)]
  mish       : allclose(rtol=1e-5, atol=0.02) both modes, over all normals —
               wider than the merged tests' atol=0.008 because the exhaustive
               sweep hits mish's curvature trough (x~-1.19)  [test_activation]

NOTE: the original 25 cases were hardware-confirmed (25/25 passed). Later
additions (row-major smoke, exp/gelu tail checks; 34 cases total) surfaced two
bad test assumptions on a hardware run — see test_row_major_layout_smoke and
test_exp_underflow docstrings — which were fixed but NOT yet re-run on
hardware. The PR description's "25/25 hardware-verified" refers only to the
original subset; the 9 cases added afterward are untested on hardware as of
this revision (no accelerator in this authoring environment; see AGENTS.md).
"""


def _assert_all_nonfinite(result, desc):
    """Every element must be non-finite (device may pack NaN as ±inf in bf16)."""
    nonfinite = ~torch.isfinite(result)
    assert nonfinite.all(), f"expected all {desc} outputs to be non-finite; {int((~nonfinite).sum())} were finite"


def _assert_finite_matches_golden(golden, result, desc):
    """Finite-input elements must produce finite output (PCC alone can hide an
    isolated NaN/Inf regression since comparison_funcs zeroes both sides)."""
    unexpected_nonfinite = torch.isfinite(golden) & ~torch.isfinite(result)
    assert (
        not unexpected_nonfinite.any()
    ), f"{desc}: {int(unexpected_nonfinite.sum())} finite-input elements produced a non-finite device output"


# ─────────────────────────────────────────────────────────────────────────────
# sqrt, rsqrt — reciprocal/root ops, ULP-based on their positive domain
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "ttnn_op, golden_fn, low, ulp_accurate",
    [
        (ttnn.sqrt, torch.sqrt, 0.0, 1),
        (ttnn.rsqrt, torch.rsqrt, SMALLEST_NORMAL_BF16, 2),
    ],
    ids=["sqrt", "rsqrt"],
)
def test_root_ops_accurate(device, ttnn_op, golden_fn, low, ulp_accurate):
    """Accurate mode: exhaustive positive-normal bf16 domain. sqrt is defined
    at 0; rsqrt diverges there, so its sweep starts at the smallest normal."""
    input_tensor = generate_bfloat16_bits_in_range(low, MAX_BF16)
    tt_in = to_tt_tensor(input_tensor, device)

    golden = golden_fn(input_tensor.float()).to(torch.bfloat16)
    result = ttnn.to_torch(ttnn_op(tt_in, fast_and_approximate_mode=False)).to(torch.bfloat16)

    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=ulp_accurate, allow_nonfinite=True)


@pytest.mark.parametrize(
    "ttnn_op, golden_fn",
    [
        (ttnn.sqrt, torch.sqrt),
        (ttnn.rsqrt, torch.rsqrt),
    ],
    ids=["sqrt", "rsqrt"],
)
def test_root_ops_fast(device, ttnn_op, golden_fn):
    """Fast mode: exhaustive bf16 values in [1, 100], ULP <= 2 (matches
    test_unary.py::test_unary_root_ops_ttnn)."""
    input_tensor = generate_bfloat16_bits_in_range(1.0, 100.0)
    tt_in = to_tt_tensor(input_tensor, device)

    golden = golden_fn(input_tensor.float()).to(torch.bfloat16)
    result = ttnn.to_torch(ttnn_op(tt_in, fast_and_approximate_mode=True)).to(torch.bfloat16)

    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=2)


@pytest.mark.parametrize(
    "ttnn_op, high",
    [
        (ttnn.sqrt, -SMALLEST_NORMAL_BF16),  # sqrt(0) = 0 is in-domain; keep 0 out of this sweep
        (ttnn.rsqrt, 0.0),  # rsqrt(0) = +inf is out-of-domain; include 0 here
    ],
    ids=["sqrt", "rsqrt"],
)
def test_root_ops_negative_domain(device, ttnn_op, high):
    """Out-of-domain must be non-finite. sqrt excludes 0 (sqrt(0)=0 is valid);
    rsqrt's sweep includes 0 (rsqrt(0)=+inf)."""
    input_tensor = generate_bfloat16_bits_in_range(-MAX_BF16, high)
    tt_in = to_tt_tensor(input_tensor, device)

    result = ttnn.to_torch(ttnn_op(tt_in))
    _assert_all_nonfinite(result, f"{ttnn_op.__name__}(x<=0)" if high == 0.0 else f"{ttnn_op.__name__}(x<0)")


@pytest.mark.parametrize(
    "ttnn_op, golden_fn, low, high, ulp",
    [
        (ttnn.sqrt, torch.sqrt, 0.0, 100.0, 1),
        (ttnn.gelu, F.gelu, 1.0, 10.0, 10),
        (ttnn.log, torch.log, 1.0, 100.0, 1),
        (ttnn.log2, torch.log2, 1.0, 100.0, 1),
        (ttnn.log10, torch.log10, 1.0, 100.0, 2),
        (ttnn.log1p, torch.log1p, 1.0, 100.0, 1),
    ],
    ids=["sqrt", "gelu", "log", "log2", "log10", "log1p"],
)
def test_row_major_layout_smoke(device, ttnn_op, golden_fn, low, high, ulp):
    """ROW_MAJOR_LAYOUT dispatch-path smoke check (accurate mode); the
    exhaustive sweeps above all use TILE_LAYOUT. gelu's range is positive-only
    ([1, 10]) to stay clear of the exp-field=1 FTZ band (see
    test_gelu_accurate) that a wider sweep would hit near zero."""
    input_tensor = generate_bfloat16_bits_in_range(low, high)
    tt_in = to_tt_tensor(input_tensor, device, layout=ttnn.ROW_MAJOR_LAYOUT)

    golden = golden_fn(input_tensor.float()).to(torch.bfloat16)
    result = ttnn.to_torch(ttnn_op(tt_in, fast_and_approximate_mode=False)).to(torch.bfloat16)

    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=ulp)


# ─────────────────────────────────────────────────────────────────────────────
# exp — ULP over its finite (non-overflow, non-underflow) range
# ─────────────────────────────────────────────────────────────────────────────


def test_exp_accurate(device):
    """Accurate mode: ULP <= 1 over [-87, 88.5], the finite range (mirrors
    test_unary_category1_bfloat16.py::test_exp_ops)."""
    input_tensor = generate_bfloat16_bits_in_range(-87.0, 88.5)
    tt_in = to_tt_tensor(input_tensor, device)

    golden = torch.exp(input_tensor.float()).to(torch.bfloat16)
    result = ttnn.to_torch(ttnn.exp(tt_in, fast_and_approximate_mode=False)).to(torch.bfloat16)

    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=1)


def test_exp_fast(device):
    """Fast mode: PCC >= 0.999 over the same finite range."""
    input_tensor = generate_bfloat16_bits_in_range(-87.0, 88.5)
    tt_in = to_tt_tensor(input_tensor, device)

    golden = torch.exp(input_tensor.float()).to(torch.bfloat16)
    result = ttnn.to_torch(ttnn.exp(tt_in, fast_and_approximate_mode=True)).to(torch.bfloat16)

    _assert_finite_matches_golden(golden, result, "exp(fast)")
    assert_with_pcc(golden, result, pcc=0.999)


def test_exp_underflow(device):
    """Underflow tail (x < -87), accurate mode only: exp(x) rounds to exactly
    0. Fast mode is excluded — hardware showed it retains a slowly-decaying
    non-zero tail well past x=-87 (up to ~1.7e-15) instead of hard-flushing,
    so only its PCC>=0.999 contract (test_exp_fast) applies there."""
    input_tensor = generate_bfloat16_bits_in_range(-MAX_BF16, -87.5)
    tt_in = to_tt_tensor(input_tensor, device)

    result = ttnn.to_torch(ttnn.exp(tt_in, fast_and_approximate_mode=False))
    assert torch.all(result == 0.0), "expected exp underflow region (x < -87) to be exactly 0 in accurate mode"


@pytest.mark.parametrize("fast", [False, True], ids=["accurate", "fast"])
def test_exp_overflow(device, fast):
    """Overflow tail (x > 88.5): exp(x) saturates to +inf, in both modes."""
    input_tensor = generate_bfloat16_bits_in_range(89.0, MAX_BF16)
    tt_in = to_tt_tensor(input_tensor, device)

    result = ttnn.to_torch(ttnn.exp(tt_in, fast_and_approximate_mode=fast))
    assert torch.all(torch.isposinf(result)), "expected exp overflow region (x > 88.5) to be +inf"


# ─────────────────────────────────────────────────────────────────────────────
# erf — bounded [-1, 1]; ULP ≤ 2 over the active band in both modes
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("fast", [False, True], ids=["accurate", "fast"])
def test_erf(device, fast):
    """erf over [-10, 10] (saturates to +-1 beyond ~+-4), ULP <= 2 both modes
    (matches test_unary_ops_ttnn.py::test_unary_erf_ttnn)."""
    input_tensor = generate_bfloat16_bits_in_range(-10.0, 10.0)
    tt_in = to_tt_tensor(input_tensor, device)

    golden = torch.erf(input_tensor.float()).to(torch.bfloat16)
    result = ttnn.to_torch(ttnn.erf(tt_in, fast_and_approximate_mode=fast)).to(torch.bfloat16)

    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=2)


# ─────────────────────────────────────────────────────────────────────────────
# gelu — accurate exhaustive (ULP ≤ 10, FTZ band excluded); fast PCC
# ─────────────────────────────────────────────────────────────────────────────


def test_gelu_accurate(device):
    """Accurate mode: all normal bf16 patterns, ULP <= 10. The exp-field=1
    band (|x| in [2^-126, 2^-125)) is excluded — there gelu(x)~=x/2 underflows
    to an fp32 subnormal that hardware DAZ/FTZ flushes to 0 (up to 128 ULP vs.
    torch), a documented artifact (see test_activation.py::test_gelu_bfloat16_accuracy)."""
    input_tensor = generate_bfloat16_bits(dtype=torch.bfloat16)  # all normals; specials→0
    tt_in = to_tt_tensor(input_tensor, device)

    golden = F.gelu(input_tensor.float()).to(torch.bfloat16)
    result = ttnn.to_torch(ttnn.gelu(tt_in, fast_and_approximate_mode=False)).to(torch.bfloat16)

    abs_x = input_tensor.abs().float()
    exp1_ftz_band = (abs_x >= 2.0**-126) & (abs_x < 2.0**-125)
    assert exp1_ftz_band.any(), "expected exp-field=1 FTZ band to be non-empty for this exhaustive sweep"

    finite_golden = torch.isfinite(golden)
    unexpected_nonfinite = finite_golden & ~exp1_ftz_band & ~torch.isfinite(result)
    assert not unexpected_nonfinite.any(), (
        f"gelu(accurate): {int(unexpected_nonfinite.sum())} finite-input elements outside the documented "
        "FTZ band produced a non-finite device output"
    )

    keep = ~exp1_ftz_band & finite_golden
    assert_with_ulp(expected_result=golden[keep], actual_result=result[keep], ulp_threshold=10)


def test_gelu_fast(device):
    """Fast mode (FastLut): PCC >= 0.999 over [-10, 10] — its golden skips
    generic comparison, so a bounded active-band PCC is the meaningful gate
    (matches test_unary_ops_ttnn.py::test_unary_gelu_ttnn)."""
    input_tensor = generate_bfloat16_bits_in_range(-10.0, 10.0)
    tt_in = to_tt_tensor(input_tensor, device)

    golden = F.gelu(input_tensor.float()).to(torch.bfloat16)
    result = ttnn.to_torch(ttnn.gelu(tt_in, fast_and_approximate_mode=True)).to(torch.bfloat16)

    _assert_finite_matches_golden(golden, result, "gelu(fast)")
    assert_with_pcc(golden, result, pcc=0.999)


# ─────────────────────────────────────────────────────────────────────────────
# log, log2, log10, log1p — ULP on their positive/(-1,∞) domain; fast allclose
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "ttnn_op, golden_fn, ulp",
    [
        (ttnn.log, torch.log, 1),
        (ttnn.log2, torch.log2, 1),
        (ttnn.log10, torch.log10, 2),
    ],
    ids=["log", "log2", "log10"],
)
def test_log_family_accurate(device, ttnn_op, golden_fn, ulp):
    """Accurate mode: exhaustive positive-normal bf16 domain. log10's extra
    base-change multiply costs a 2nd ULP vs. log/log2 (see test_math.py).
    x <= 0 is out-of-domain, covered separately."""
    input_tensor = generate_bfloat16_bits_in_range(SMALLEST_NORMAL_BF16, MAX_BF16)
    tt_in = to_tt_tensor(input_tensor, device)

    golden = golden_fn(input_tensor.float()).to(torch.bfloat16)
    result = ttnn.to_torch(ttnn_op(tt_in, fast_and_approximate_mode=False)).to(torch.bfloat16)

    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=ulp, allow_nonfinite=True)


@pytest.mark.parametrize("ttnn_op", [ttnn.log, ttnn.log2, ttnn.log10], ids=["log", "log2", "log10"])
def test_log_family_nonpositive_domain(device, ttnn_op):
    """Out-of-domain (x <= 0): log(0)=-inf, log(x<0)=NaN, so output must be
    non-finite."""
    input_tensor = generate_bfloat16_bits_in_range(-MAX_BF16, 0.0)
    tt_in = to_tt_tensor(input_tensor, device)

    result = ttnn.to_torch(ttnn_op(tt_in))
    _assert_all_nonfinite(result, f"{ttnn_op.__name__}(x<=0)")


def test_log1p_accurate(device):
    """Accurate mode: ULP <= 1 for x > -1; x <= -1 must be non-finite
    (log1p(-1)=-inf, log1p(x<-1)=NaN). See test_unary_ops_ttnn.py::test_unary_log1p_ttnn."""
    input_tensor = generate_bfloat16_bits(dtype=torch.bfloat16)  # all normals; specials→0
    tt_in = to_tt_tensor(input_tensor, device)

    golden = torch.log1p(input_tensor.float()).to(torch.bfloat16)
    result = ttnn.to_torch(ttnn.log1p(tt_in, fast_and_approximate_mode=False)).to(torch.bfloat16)

    x = input_tensor.float()
    in_domain = x > -1.0
    assert_with_ulp(
        expected_result=golden[in_domain], actual_result=result[in_domain], ulp_threshold=1, allow_nonfinite=True
    )

    out_of_domain = x <= -1.0
    assert out_of_domain.any(), "expected x <= -1 band to be non-empty for this exhaustive sweep"
    _assert_all_nonfinite(result[out_of_domain], "log1p(x<=-1)")


@pytest.mark.parametrize(
    "ttnn_op", [ttnn.log, ttnn.log2, ttnn.log10, ttnn.log1p], ids=["log", "log2", "log10", "log1p"]
)
def test_log_family_fast(device, ttnn_op):
    """Fast mode: allclose(atol=0.0625) over [1, 100] (matches
    test_unary_ops_ttnn.py::test_unary_log_like_fast_approx_ttnn)."""
    input_tensor = generate_bfloat16_bits_in_range(1.0, 100.0)
    tt_in = to_tt_tensor(input_tensor, device)

    golden_fn = {ttnn.log: torch.log, ttnn.log2: torch.log2, ttnn.log10: torch.log10, ttnn.log1p: torch.log1p}[ttnn_op]
    golden = golden_fn(input_tensor.float()).to(torch.bfloat16)
    result = ttnn.to_torch(ttnn_op(tt_in, fast_and_approximate_mode=True)).to(torch.bfloat16)

    assert_allclose(result, golden, atol=0.0625)


# ─────────────────────────────────────────────────────────────────────────────
# mish — x * tanh(softplus(x)); compound, allclose in both modes
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("fast", [False, True], ids=["accurate", "fast"])
def test_mish(device, fast):
    """mish over all normal bf16 values, allclose(rtol=1e-5, atol=0.02) in both
    modes. atol is 0.02 rather than the merged tests' 0.008 (test_unary.py::
    test_unary_mish, [-20, 100]) because the exhaustive sweep hits mish's
    curvature trough near x~-1.19, where the SFPU chain deviates by up to
    0.0156 (hardware-observed) — still a meaningful gate vs. an identity
    kernel's ~0.88 deviation there."""
    input_tensor = generate_bfloat16_bits(dtype=torch.bfloat16)  # all normals; specials→0
    tt_in = to_tt_tensor(input_tensor, device)

    golden = F.mish(input_tensor.float()).to(torch.bfloat16)
    result = ttnn.to_torch(ttnn.mish(tt_in, fast_and_approximate_mode=fast)).to(torch.bfloat16)

    assert_allclose(result, golden, rtol=1e-5, atol=0.02)
