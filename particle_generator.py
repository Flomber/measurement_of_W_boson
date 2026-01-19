"""Generate pp -> W± -> mu nu events with Pythia8 and write JSON.

Run with a simple command:
    python particle_generator.py

Adjust the CONFIG values below to change event count, energy, seed, or output path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple


@dataclass
class Config:
    events: int = 100_000
    energy_tev: float = 13.0
    seed: int | None = None
    output: Path = Path("data/wmunu_events.json")


def init_pythia(cfg: Config):
    import pythia8  # Local import to avoid import errors when Pythia8 is absent

    pythia = pythia8.Pythia()

    # Beam setup: pp collisions at the requested sqrt(s).
    pythia.readString("Beams:idA = 2212")
    pythia.readString("Beams:idB = 2212")
    pythia.readString(f"Beams:eCM = {cfg.energy_tev * 1000.0}")

    # Enable single W production (both charges).
    pythia.readString("WeakSingleBoson:ffbar2W = on")

    # Force W decays to muon + corresponding neutrino.
    pythia.readString("24:onMode = off")
    pythia.readString("24:onIfMatch = 13 -14")  # W+ -> mu+ nu_mu-bar
    pythia.readString("-24:onMode = off")
    pythia.readString("-24:onIfMatch = -13 14")  # W- -> mu- nu_mu

    if cfg.seed is not None:
        pythia.readString(f"Random:seed = {cfg.seed}")
        pythia.readString("Random:setSeed = on")

    pythia.init()
    return pythia


def extract_mu_nu(event) -> Tuple[Dict[str, float], Dict[str, float]] | None:
    """Return final-state muon and neutrino 4-vectors if present."""

    mu = None
    nu = None
    for p in event:
        if not p.isFinal():
            continue
        pid = p.id()
        if abs(pid) == 13 and mu is None:
            mu = {"px": p.px(), "py": p.py(), "pz": p.pz(), "E": p.e(), "id": pid}
        elif abs(pid) == 14 and nu is None:
            nu = {"px": p.px(), "py": p.py(), "pz": p.pz(), "E": p.e(), "id": pid}
        if mu and nu:
            break

    if mu and nu:
        return mu, nu
    return None


def generate_events(cfg: Config):
    pythia = init_pythia(cfg)

    events: List[Dict[str, Any]] = []
    for i_evt in range(cfg.events):
        if not pythia.next():
            continue

        pair = extract_mu_nu(pythia.event)
        if pair is None:
            continue

        mu, nu = pair
        events.append({"event": i_evt, "muon": mu, "neutrino": nu})

    meta = {
        "generator": "pythia8",
        "version": getattr(pythia, "__version__", "unknown"),
        "process": "pp -> W+/W- -> mu nu",
        "sqrt_s_GeV": cfg.energy_tev * 1000.0,
        "events_requested": cfg.events,
        "events_stored": len(events),
    }

    payload = {"meta": meta, "events": events}
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    with cfg.output.open("w", encoding="utf-8") as f:
        json.dump(payload, f)


def main():
    cfg = Config()
    generate_events(cfg)


if __name__ == "__main__":
    main()