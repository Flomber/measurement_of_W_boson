"""Generate pp -> W± -> mu nu events with Pythia8 and write JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def init_pythia(energy_tev: float = 13.0, seed: int | None = None):
    import pythia8  # Imported inside to avoid ImportError on non-Pythia hosts

    pythia = pythia8.Pythia()

    # Beam setup: pp collisions at the requested sqrt(s).
    pythia.readString("Beams:idA = 2212")
    pythia.readString("Beams:idB = 2212")
    pythia.readString(f"Beams:eCM = {energy_tev * 1000.0}")

    # Enable single W production (both charges).
    pythia.readString("WeakSingleBoson:ffbar2W = on")

    # Force W decays to muon + corresponding neutrino.
    pythia.readString("24:onMode = off")
    pythia.readString("24:onIfMatch = 13 -14")  # W+ -> mu+ nu_mu-bar
    pythia.readString("-24:onMode = off")
    pythia.readString("-24:onIfMatch = -13 14")  # W- -> mu- nu_mu

    # Optional: set a deterministic random seed for reproducibility.
    if seed is not None:
        pythia.readString(f"Random:seed = {seed}")
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


def generate_events(num_events: int, energy_tev: float, seed: int | None, out_path: Path):
    pythia = init_pythia(energy_tev=energy_tev, seed=seed)

    events: List[Dict[str, Any]] = []
    for i_evt in range(num_events):
        if not pythia.next():
            continue  # Skip failed event generation

        pair = extract_mu_nu(pythia.event)
        if pair is None:
            continue

        mu, nu = pair
        events.append({"event": i_evt, "muon": mu, "neutrino": nu})

    # Build a small metadata header.
    meta = {
        "generator": "pythia8",
        "version": getattr(pythia, "__version__", "unknown"),
        "process": "pp -> W+/W- -> mu nu",
        "sqrt_s_GeV": energy_tev * 1000.0,
        "events_requested": num_events,
        "events_stored": len(events),
    }

    payload = {"meta": meta, "events": events}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate W->mu nu events with Pythia8")
    parser.add_argument("--events", type=int, default=100_000, help="Number of events to generate")
    parser.add_argument("--energy-tev", type=float, default=13.0, help="Center-of-mass energy in TeV")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (None for random)")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/wmunu_events.json"),
        help="Output JSON path",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    generate_events(num_events=args.events, energy_tev=args.energy_tev, seed=args.seed, out_path=args.output)


if __name__ == "__main__":
    main()
