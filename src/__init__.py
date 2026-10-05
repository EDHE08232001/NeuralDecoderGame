"""qec-arena: a neural QEC decoder game (Qiskit Fall Fest 2026 project).

Modules (see README.md and the project plan, section 7):

    noise       noise-model specifications + IBM device calibration -> Stim noise
    circuits    Stim and Qiskit repetition-code / surface-code circuit builders
    data        sampling syndromes, dataset save/load
    decoders/   mwpm (PyMatching), greedy (human-like), neural (PyTorch)
    evaluate    logical-error-rate sweeps, confidence intervals, analytic checks
    plotting    the four figures of the evaluation protocol
    experiments E1-E5 pipelines used by scripts/ and notebooks/
    hardware    IBM Runtime (PINQ2) job submission + offline Aer fallback
"""

__version__ = "1.0.0"
