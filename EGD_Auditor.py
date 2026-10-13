#!/usr/bin/env python3
"""EGD/EGD-QEC research auditor.

This tool is deliberately conservative. It checks exact finite instances of the
EGD halving construction and emits a proof-obligation report. It does NOT claim
that finite tests prove asymptotic efficiency, AC^0[qpoly] membership, or P != NP.

Custom status convention requested by the project:
    printed `Exit code: 1` = end-to-end EGD claim validated
    printed `Exit code: 0` = not validated / at least one critical obligation open
The operating-system process exit status follows the same custom convention.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Sequence

VERSION = "0.1.0"


@dataclass
class Check:
    key: str
    status: str  # PROVED_FROM_DEFINITIONS, TEST_PASS, OPEN, NOT_IMPLEMENTED, etc.
    summary: str
    evidence: str
    blocks_validation: bool = False


@dataclass
class HalvingStep:
    depth: int
    prefix_before: str
    count_zero: int
    count_one: int
    chosen_bit: str
    count_after: int
    phi_before: str
    phi_zero: str
    phi_one: str
    phi_after: str


@dataclass
class AuditReport:
    auditor_version: str
    generated_utc: str
    python_version: str
    parameters: dict[str, Any]
    finite_tests: dict[str, Any]
    math_checks: list[dict[str, Any]]
    proof_obligations: list[dict[str, Any]]
    verdict: str
    exit_code_custom: int
    disclaimer: str


def frac_text(x: Fraction) -> str:
    return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"


def count_compatible(candidate_tables: Sequence[str], prefix: str) -> int:
    """Count descriptions/tables compatible with prefix; duplicate strings count separately."""
    if not all(set(word) <= {"0", "1"} for word in candidate_tables):
        raise ValueError("Candidate tables must contain only 0/1 characters.")
    if candidate_tables:
        width = len(candidate_tables[0])
        if any(len(word) != width for word in candidate_tables):
            raise ValueError("All candidate tables must have the same width.")
        if len(prefix) > width:
            raise ValueError("Prefix is longer than the candidate truth tables.")
    if set(prefix) - {"0", "1"}:
        raise ValueError("Prefix must contain only 0/1 characters.")
    return sum(word.startswith(prefix) for word in candidate_tables)


def potential(candidate_tables: Sequence[str], prefix: str, r: int) -> Fraction:
    """Phi(u) = A(u) / 2^(r-|u|), evaluated exactly as a rational."""
    if r < len(prefix):
        raise ValueError("r must be at least the prefix length.")
    if not candidate_tables:
        return Fraction(0, 2 ** (r - len(prefix)))
    if len(prefix) > len(candidate_tables[0]):
        raise ValueError("Prefix is longer than the candidate truth tables.")
    return Fraction(count_compatible(candidate_tables, prefix), 2 ** (r - len(prefix)))


def exact_halving(candidate_tables: Sequence[str], r: int) -> tuple[str, list[HalvingStep]]:
    """Run exact EGD halving on an explicit finite candidate family.

    Each list entry is treated as a circuit description. Duplicate truth tables
    are permitted because different circuit descriptions may compute the same
    Boolean function. This is an exponential/reference implementation, not the
    proposed efficient EGD-QEC implementation.
    """
    if not candidate_tables:
        raise ValueError("Use a nonempty candidate family for this audit.")
    if r < 1:
        raise ValueError("r must be >= 1.")
    if any(set(word) - {"0", "1"} for word in candidate_tables):
        raise ValueError("Candidate tables must contain only 0/1 characters.")
    width = len(candidate_tables[0])
    if any(len(word) != width for word in candidate_tables):
        raise ValueError("All candidate tables must have the same width.")
    if r > width:
        raise ValueError("Need r <= truth-table width to construct a prefix.")
    if (1 << r) <= len(candidate_tables):
        raise ValueError("Need 2**r > number of circuit descriptions for the halving guarantee.")

    prefix = ""
    trace: list[HalvingStep] = []
    for i in range(r):
        a0 = count_compatible(candidate_tables, prefix + "0")
        a1 = count_compatible(candidate_tables, prefix + "1")
        # Deterministic tie breaking: choose 0 on ties.
        chosen = "0" if a0 <= a1 else "1"
        before = count_compatible(candidate_tables, prefix)
        phi_before = potential(candidate_tables, prefix, r)
        phi0 = potential(candidate_tables, prefix + "0", r)
        phi1 = potential(candidate_tables, prefix + "1", r)
        prefix += chosen
        after = count_compatible(candidate_tables, prefix)
        phi_after = potential(candidate_tables, prefix, r)

        if before != a0 + a1:
            raise AssertionError("Partition identity A(u)=A(u0)+A(u1) failed.")
        if phi_before != (phi0 + phi1) / 2:
            raise AssertionError("Potential conservation identity failed.")
        if after > before // 2:
            raise AssertionError("Exact halving did not choose a child of minimum count.")

        trace.append(HalvingStep(
            depth=i + 1,
            prefix_before=prefix[:-1],
            count_zero=a0,
            count_one=a1,
            chosen_bit=chosen,
            count_after=after,
            phi_before=frac_text(phi_before),
            phi_zero=frac_text(phi0),
            phi_one=frac_text(phi1),
            phi_after=frac_text(phi_after),
        ))

    final_count = count_compatible(candidate_tables, prefix)
    if final_count != 0:
        raise AssertionError("The theorem's terminal conclusion failed on this instance.")
    return prefix, trace


def greedy_step_lemma_holds(phi0: Fraction, phi1: Fraction,
                            estimate0: Fraction, estimate1: Fraction,
                            epsilon: Fraction) -> bool:
    """Check one instance of the approximate greedy-step inequality.

    Assumes both estimates are within epsilon of their corresponding true child
    potentials and the selected child is the one with minimum estimate.
    """
    if epsilon < 0:
        raise ValueError("epsilon must be nonnegative.")
    if abs(estimate0 - phi0) > epsilon or abs(estimate1 - phi1) > epsilon:
        raise ValueError("Supplied estimates do not satisfy the claimed error bound.")
    chosen = phi0 if estimate0 <= estimate1 else phi1
    parent = (phi0 + phi1) / 2
    return chosen <= parent + epsilon


def accumulated_error_suffices(phi_initial: Fraction, epsilons: Sequence[Fraction]) -> bool:
    """Check the strict sufficient condition Phi(empty)+sum eps_i < 1."""
    if phi_initial < 0 or any(e < 0 for e in epsilons):
        raise ValueError("Potentials and error tolerances must be nonnegative.")
    return phi_initial + sum(epsilons, Fraction(0)) < 1


def exact_count_recovered_by_rounding(count: int, r: int, depth: int,
                                      estimate_phi: Fraction,
                                      eta: Fraction) -> bool:
    """Check terminal/partial count rounding under strict granularity condition."""
    if not (0 <= depth <= r):
        raise ValueError("depth must be between 0 and r.")
    if eta < 0:
        raise ValueError("eta must be nonnegative.")
    true_phi = Fraction(count, 2 ** (r - depth))
    if abs(estimate_phi - true_phi) > eta:
        raise ValueError("estimate violates the supplied error bound")
    if eta >= Fraction(1, 2 ** (r - depth + 1)):
        return False
    scaled = estimate_phi * (2 ** (r - depth))
    # Exact nearest integer for nonnegative values, with ties rounded upward.
    rounded = (2 * scaled.numerator + scaled.denominator) // (2 * scaled.denominator)
    return rounded == count


def granularity_barrier_holds(M: int, r: int, epsilons: Sequence[Fraction]) -> bool:
    """Check the implication in the EGD additive-error granularity lemma.

    If M>=1, 2**r>M, and M/2**r + sum(eps_i)<1, then some depth i in
    1..r has eps_i < 2**(i-r-1); at that depth a uniformly valid estimate
    has precision sufficient to recover every integer survivor count by
    rounding. Returns True when the implication holds (including vacuous cases).
    """
    if M < 1 or r < 1:
        raise ValueError("M and r must be positive")
    if len(epsilons) != r:
        raise ValueError("Provide exactly r error tolerances, one per depth")
    if any(e < 0 for e in epsilons):
        raise ValueError("Error tolerances must be nonnegative")
    condition = Fraction(M, 2 ** r) + sum(epsilons, Fraction(0)) < 1
    if not condition:
        return True
    thresholds = [Fraction(1, 2 ** (r - i + 1)) for i in range(1, r + 1)]
    return any(e < threshold for e, threshold in zip(epsilons, thresholds))


def circuit_description_upper_bound(input_bits: int, size: int) -> tuple[int, int]:
    """Conservative counting bound for topologically ordered fan-in-2 circuits.

    Basis: AND, OR, NOT; each gate may select two predecessor wires from at most
    input_bits + size + 1 possible wires (overcounting NOT's one input); output
    wire also has at most input_bits + size + 1 choices. For k<=size gates,
    choices are bounded by (4*(input_bits+size+1)^2)^size. Multiplying by the
    number of possible k values gives B. Returns (B, ceil(log2(B))).

    This bound is specific to the stated toy encoding model; other circuit models
    require their own encoding/counting argument.
    """
    if input_bits < 1 or size < 0:
        raise ValueError("input_bits must be >= 1 and size >= 0")
    wires = input_bits + size + 1
    base = 4 * wires * wires
    bound = (size + 1) * wires * (base ** size)
    ceil_log2_bound = (bound - 1).bit_length()  # exact ceil(log2(bound))
    return bound, ceil_log2_bound


def required_rounds_from_bound(description_bound: int) -> int:
    """Return r with 2**r > description_bound, exactly."""
    if description_bound < 1:
        raise ValueError("description_bound must be >=1")
    return description_bound.bit_length()


def run_finite_tests(seed: int = 7321, trials: int = 500) -> dict[str, Any]:
    """Run reproducible small-instance tests; these are sanity checks, not proofs."""
    rng = random.Random(seed)
    identity_cases = 0
    exact_halving_cases = 0
    greedy_lemma_cases = 0
    error_budget_cases = 0

    # Exhaustive candidate families from small truth tables (duplicates allowed).
    all_words = [format(i, "03b") for i in range(8)]
    for _ in range(100):
        m = rng.randint(1, 7)
        family = [rng.choice(all_words) for _ in range(m)]
        r = m.bit_length()
        if r <= 3 and (1 << r) > m:
            prefix, trace = exact_halving(family, r)
            assert count_compatible(family, prefix) == 0
            exact_halving_cases += 1
            for step in trace:
                # The exact trace itself records and verifies the parent/children identity.
                assert step.count_after == min(step.count_zero, step.count_one)
        for depth in range(0, min(r, 3)):
            u = "0" * depth
            parent = potential(family, u, r)
            p0 = potential(family, u + "0", r)
            p1 = potential(family, u + "1", r)
            assert parent == (p0 + p1) / 2
            identity_cases += 1

    # Exhaustively exercise the approximate greedy inequality over rational grid points.
    for _ in range(trials):
        c0 = Fraction(rng.randint(0, 32), 16)
        c1 = Fraction(rng.randint(0, 32), 16)
        eps = Fraction(rng.randint(0, 8), 16)
        # Select each estimate's error independently within [-eps, eps].
        # Use a common denominator to keep all values rational.
        denom = 16
        e0 = Fraction(rng.randint(-int(eps * denom), int(eps * denom)), denom)
        e1 = Fraction(rng.randint(-int(eps * denom), int(eps * denom)), denom)
        assert greedy_step_lemma_holds(c0, c1, c0 + e0, c1 + e1, eps)
        greedy_lemma_cases += 1

        phi_initial = Fraction(rng.randint(0, 8), 16)
        eps_list = [Fraction(rng.randint(0, 2), 64) for _ in range(4)]
        condition = accumulated_error_suffices(phi_initial, eps_list)
        if condition:
            error_budget_cases += 1

    # Granularity-barrier implication: M=3, r=3 and half-threshold tolerances.
    test_eps = [Fraction(1, 2 ** (3 - i + 2)) for i in range(1, 4)]
    assert granularity_barrier_holds(3, 3, test_eps)

    # Known edge cases.
    bnd, logb = circuit_description_upper_bound(2, 1)
    assert bnd >= 1 and 2 ** logb >= bnd
    assert 2 ** required_rounds_from_bound(bnd) > bnd

    return {
        "seed": seed,
        "randomized_trials": trials,
        "potential_identity_instances": identity_cases,
        "exact_halving_instances": exact_halving_cases,
        "approximate_greedy_lemma_instances": greedy_lemma_cases,
        "error_budget_sufficient_instances": error_budget_cases,
        "all_finite_tests_passed": True,
        "interpretation": "Finite implementation checks only; they do not prove asymptotic complexity or class membership.",
    }


def make_proof_obligations() -> list[Check]:
    """Evidence ledger: keeps proved combinatorics separate from open algorithms."""
    return [
        Check(
            "MATH-01-PARTITION",
            "PROVED_FROM_DEFINITIONS",
            "A(u)=A(u0)+A(u1), and Phi(u)=(Phi(u0)+Phi(u1))/2.",
            "Disjoint partition by the next truth-table bit; also checked in finite tests.",
        ),
        Check(
            "MATH-02-EXACT-HALVING",
            "PROVED_FROM_DEFINITIONS",
            "If 2^r exceeds the number M of circuit descriptions and r is within the truth-table length, exact halving yields a prefix with zero survivors.",
            "At each step A(next)<=A(current)/2, hence A(u_r)<=M/2^r<1; A(u_r) is a nonnegative integer.",
        ),
        Check(
            "MATH-03-APPROXIMATE-STEP",
            "PROVED_FROM_DEFINITIONS",
            "Choosing the child with lower estimated potential, with both absolute errors <= epsilon_i, gives Phi(child)<=Phi(parent)+epsilon_i.",
            "Two-case proof from the exact mean identity and the ordering of estimates; rational finite tests included.",
        ),
        Check(
            "MATH-05-GRANULARITY-BARRIER",
            "PROVED_FROM_DEFINITIONS",
            "Under a strict additive error budget, at least one depth has precision fine enough to recover exact survivor counts for all prefixes at that depth.",
            "The depth thresholds sum to 1-2^-r, while the budget is strictly below 1-M/2^r <= 1-2^-r for M>=1.",
        ),
        Check(
            "MATH-04-ERROR-BUDGET",
            "PROVED_FROM_DEFINITIONS",
            "Phi(empty)+sum(epsilon_i)<1 implies the final survivor count is zero, when the stepwise error bound is valid.",
            "Iterate the approximate-step inequality and use integrality at depth r.",
        ),
        Check(
            "ALG-01-EGD-QEC-ESTIMATOR",
            "NOT_IMPLEMENTED",
            "No concrete syndrome/relaxation algorithm mapping a prefix to an estimate is supplied in this project bundle.",
            "A named architecture is not executable evidence. Need algorithm, error theorem, and worst-case bit-complexity analysis.",
            True,
        ),
        Check(
            "ALG-02-WORST-CASE-EFFICIENCY",
            "OPEN",
            "No proof shows that every estimate is computed within the required worst-case resources.",
            "The explicit reference halving implementation scans a finite candidate family and is exponential in the description length in the worst case.",
            True,
        ),
        Check(
            "ALG-03-ERROR-COMPOSITION",
            "OPEN",
            "QEC/channel error assumptions, estimator bias, numerical error, and adaptive composition are not fully specified or proved.",
            "Hoeffding/repetition bounds only address specified random channel errors; they do not by themselves certify the potential estimate.",
            True,
        ),
        Check(
            "ST-01-ABSOLUTE-PROTOCOL",
            "NOT_IMPLEMENTED",
            "No complete Student-Teacher protocol using the EGD state transition is implemented with a constant consultation budget.",
            "The exact halving simulator does not itself provide the required constant-query interface.",
            True,
        ),
        Check(
            "ST-02-AC0-QPOLY-MEMBERSHIP",
            "OPEN",
            "Membership of one absolute EGD protocol family in the required polylog-uniform AC^0[qpoly] class is not established.",
            "Python tests or benchmark timing cannot prove nonuniform circuit-class membership or the requested uniformity theorem.",
            True,
        ),
        Check(
            "ST-03-INFINITELY-MANY-EXONENTS",
            "OPEN",
            "A single protocol with the required guarantees for infinitely many exponents b has not been proved.",
            "A separate construction for each fixed b does not automatically establish the absolute protocol condition.",
            True,
        ),
        Check(
            "CONCLUSION-01-P-NE-NP",
            "CONDITIONAL_ONLY",
            "P != NP is not derived by this auditor.",
            "The conclusion follows only if all hypotheses of the cited Carmosino-Grosser implication are separately established.",
            True,
        ),
    ]


def make_report(input_bits: int, b: int, seed: int, trials: int) -> AuditReport:
    from datetime import datetime, timezone

    finite = run_finite_tests(seed=seed, trials=trials)
    size = input_bits ** b
    upper_bound, L = circuit_description_upper_bound(input_bits, size)
    r = L + 1  # Conservative: 2^r exceeds the encoding-count upper bound.
    N = 1 << input_bits
    fits = r <= N
    checks = [
        {
            "key": "PARAM-01-DESCRIPTION-BOUND",
            "status": "PASS" if fits else "NOT_APPLICABLE_FOR_PARAMETERS",
            "summary": ("Computed an explicit description-count upper bound, and the required prefix fits in the truth table for this instance." if fits else "The description-count bound was computed, but the sufficient prefix length exceeds the truth-table length for this small parameter instance."),
            "evidence": {
                "input_bits_n": input_bits,
                "size_bound_s=n^b": size,
                "description_count_upper_bound_B": str(upper_bound),
                "ceil_log2_B": L,
                "chosen_r=L+1": r,
                "truth_table_length_N=2^n": N,
                "r_le_N_for_this_instance": fits,
            },
        },
        {
            "key": "TEST-01-FINITE-CHECKS",
            "status": "PASS" if finite["all_finite_tests_passed"] else "FAIL",
            "summary": "Finite/reference tests of the core combinatorial identities.",
            "evidence": finite,
        },
    ]
    obligations = make_proof_obligations()
    blocking_open = [o for o in obligations if o.blocks_validation and o.status not in {"PROVED_FROM_DEFINITIONS", "TEST_PASS"}]
    validated = len(blocking_open) == 0 and fits and finite["all_finite_tests_passed"]
    return AuditReport(
        auditor_version=VERSION,
        generated_utc=datetime.now(timezone.utc).isoformat(),
        python_version=platform.python_version(),
        parameters={
            "input_bits_n": input_bits,
            "exponent_b": b,
            "truth_table_length_N": N,
            "circuit_size_s=n^b": size,
            "description_count_upper_bound_B": str(upper_bound),
            "L=ceil(log2(B))": L,
            "chosen_r": r,
            "r_le_N": fits,
        },
        finite_tests=finite,
        math_checks=checks,
        proof_obligations=[asdict(o) for o in obligations],
        verdict="VALIDATED" if validated else "NOT VALIDATED",
        exit_code_custom=1 if validated else 0,
        disclaimer=(
            "This auditor verifies reference calculations and tracks mathematical proof obligations. "
            "It cannot infer a worst-case asymptotic bound, AC^0[qpoly] membership, polylog-uniformity, "
            "or P != NP from finite tests. With the EGD-QEC implementation and absolute protocol missing, "
            "NOT VALIDATED means the claims are not established, not that EGD is impossible."
        ),
    )


def render_text_report(report: AuditReport) -> str:
    lines: list[str] = []
    lines.append("LUNZ RESEARCH GROUP — EGD / EGD-QEC AUDIT REPORT")
    lines.append("=" * 66)
    lines.append(f"Auditor version: {report.auditor_version}")
    lines.append(f"Generated (UTC): {report.generated_utc}")
    lines.append(f"Python: {report.python_version}")
    lines.append("")
    lines.append("1. PARAMETERS AND FINITE MODEL")
    for k, v in report.parameters.items():
        lines.append(f"   {k}: {v}")
    lines.append("")
    lines.append("2. MACHINE-CHECKED FINITE TESTS")
    for k, v in report.finite_tests.items():
        lines.append(f"   {k}: {v}")
    lines.append("")
    lines.append("3. MATHEMATICAL CHECKS")
    for item in report.math_checks:
        lines.append(f"   [{item['status']}] {item['key']}: {item['summary']}")
        lines.append(f"      Evidence: {item['evidence']}")
    lines.append("")
    lines.append("4. FORMAL PROOF-OBLIGATION LEDGER")
    for item in report.proof_obligations:
        tag = "BLOCKS VALIDATION" if item["blocks_validation"] else "NON-BLOCKING"
        lines.append(f"   [{item['status']}] {item['key']} ({tag})")
        lines.append(f"      {item['summary']}")
        lines.append(f"      Evidence / gap: {item['evidence']}")
    lines.append("")
    lines.append("5. FORMAL CONCLUSION")
    lines.append(f"   Verdict: {report.verdict}")
    lines.append("   The exact-halving mathematical argument is valid under its stated assumptions.")
    lines.append("   The additive-error granularity lemma also flags a precision/counting barrier for uniformly accurate estimators.")
    lines.append("   The efficient EGD-QEC estimator, error composition, absolute Student-Teacher protocol,")
    lines.append("   AC^0[qpoly] membership, polylog-uniformity and infinite-exponent condition are NOT established.")
    lines.append("   Therefore this audit does not certify that EGD meets the Carmosino-Grosser hypotheses.")
    lines.append("   This is not a proof that EGD cannot work; it is a finding that the evidence is incomplete.")
    lines.append("")
    lines.append("6. EXIT STATUS (CUSTOM PROJECT CONVENTION)")
    lines.append("   Exit code: " + str(report.exit_code_custom))
    lines.append("   Convention: 1 = all end-to-end claims validated; 0 = not validated.")
    lines.append("")
    lines.append("DISCLAIMER")
    lines.append("   " + report.disclaimer)
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Conservative EGD/EGD-QEC mathematics and proof-obligation auditor")
    parser.add_argument("--n", type=int, default=8, help="number of Boolean input bits for the illustrative circuit-size bound (default: 4)")
    parser.add_argument("--b", type=int, default=1, help="fixed exponent for the illustrative bound (default: 1)")
    parser.add_argument("--seed", type=int, default=7321, help="deterministic random seed for finite tests")
    parser.add_argument("--trials", type=int, default=500, help="number of rational-grid checks for approximate greedy lemma")
    parser.add_argument("--json-out", type=Path, default=None, help="optional path to write machine-readable JSON report")
    parser.add_argument("--text-out", type=Path, default=None, help="optional path to write this formal text report")
    args = parser.parse_args(argv)
    if args.n < 1 or args.b < 1:
        parser.error("--n and --b must both be >= 1")
    if args.trials < 1:
        parser.error("--trials must be >= 1")

    started = time.perf_counter()
    report = make_report(args.n, args.b, args.seed, args.trials)
    elapsed = time.perf_counter() - started
    # Include runtime as an observation, never as an asymptotic proof.
    report.math_checks.append({
        "key": "RUNTIME-OBSERVATION",
        "status": "OBSERVED",
        "summary": "Wall-clock runtime of this finite audit run (not an asymptotic guarantee).",
        "evidence": {"seconds": round(elapsed, 6)},
    })
    text_report = render_text_report(report)
    print(text_report, end="")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(asdict(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.text_out:
        args.text_out.parent.mkdir(parents=True, exist_ok=True)
        args.text_out.write_text(text_report, encoding="utf-8")

    # Custom convention explicitly requested by the user: 1 validated, 0 not validated.
    # This means the OS process code is also 0 in the current, not-yet-validated state.
    return report.exit_code_custom


if __name__ == "__main__":
    sys.exit(main())
