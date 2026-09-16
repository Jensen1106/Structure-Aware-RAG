"""Build Task 3 dataset — open-ended maintainability design suggestions.

20 equipment scenarios across 4 equipment families:
  - aviation (aircraft engine, avionics, flight control)
  - naval    (power cabinet, radar, fire suppression)
  - armored  (vehicle electronics, battlefield repair, armor)
  - electronics (radar, comm, datalink, LRU)

Each scenario requires the model to generate N = 3~5 design suggestions
referencing standard clauses and reference values.

Hand-curated templates (no LLM needed, fully reproducible, cheap).

Output: data/maintqa/task3_open_generation.jsonl
"""

from __future__ import annotations
import json
from pathlib import Path

OUT_FILE = Path("data/maintqa/task3_open_generation.jsonl")


SCENARIOS = [
    # === Aviation (5) ===
    ("aviation", "Accessory Bay Layout Design for a Turbofan Engine",
     "Design maintainability requirements for the engine accessory bay (oil filter, fuel pump, ignition box, etc.), focusing on field-level rapid disassembly/assembly, accessibility, and tool clearance.",
     4),
    ("aviation", "Airborne Avionics LRU Rack Design",
     "Define maintainability requirements for airborne avionics equipment rack, covering interchangeability, accessibility, fault isolation, and LRU weight.",
     4),
    ("aviation", "Flight Control System Hydraulic Actuator Maintainability Design",
     "Maintainability scheme for flight control hydraulic servo actuators, addressing high-pressure oil circuit safety, seal replacement, and disassembly/assembly tool force.",
     3),
    ("aviation", "Aircraft Landing Gear Maintainability Design",
     "Maintenance design for landing gear retraction/extension mechanisms, tires, and brakes, focusing on manual operation accessibility, visual inspection, and torque limits.",
     4),
    ("aviation", "Cockpit Instrument Panel Replacement Design",
     "Quick-change panel design for cockpit instrument panels, focusing on error-proof identification, standard connectors, and single-person operation.",
     3),
    # === Naval (5) ===
    ("naval", "Naval Main Propulsion Diesel Engine Maintainability Design",
     "Field-level maintainability design for marine medium-speed diesel engines, focusing on disassembly/assembly access, engine room lighting, and lifting capacity.",
     4),
    ("naval", "Shipboard Radar Array Cooling Module Maintenance",
     "Maintenance scheme for TR module cooling circuits of shipboard active phased array radar, focusing on module interchangeability, tool access channels, and test points.",
     4),
    ("naval", "Naval Power Distribution Cabinet Maintainability Design",
     "Maintainability requirements for power distribution cabinets, addressing high-voltage live-line work safety, interlocks, and accessibility.",
     3),
    ("naval", "Shipboard Fire Suppression System Maintainability Design",
     "Maintenance scheme for CO2/foam fire suppression piping and cylinder banks, focusing on cylinder weighing, pipeline leak detection, and maintenance safety.",
     3),
    ("naval", "Hull Inter-Compartment Cable Penetration Maintenance",
     "Maintainability scheme for watertight cable penetrations, focusing on watertight integrity, labeling, and tool accessibility.",
     3),
    # === Armored (5) ===
    ("armored", "Armored Vehicle Powerpack Bay Accessibility Design",
     "Maintainability design for the powerpack bay of main battle tanks / tracked infantry fighting vehicles, focusing on rear deck opening, engine integral lifting, and replacement time.",
     4),
    ("armored", "Battlefield Maintenance Quick-Change Assembly Design",
     "LRU / assembly quick-change design under field conditions (transmission / engine / running gear), focusing on tool-free operation and single-soldier execution.",
     4),
    ("armored", "Vehicle-Mounted Electronics (Fire Control/Communication) Redundancy Design",
     "Maintainability and redundancy design for vehicle-mounted fire control computers and communication radios, focusing on error-proof connectors, hot swapping, and fault detection.",
     3),
    ("armored", "Armor Protection Plate Inspection Access Design",
     "Accessibility design for vehicle-mounted equipment behind removable armor plates, focusing on opening dimensions, latches, and two-person work space.",
     3),
    ("armored", "Vehicle Running Gear Maintenance Design",
     "Maintainability design for drive sprockets / idler wheels / tracks, focusing on high-temperature conditions, heavy object handling, and preventive maintenance intervals.",
     4),
    # === Electronics (5) ===
    ("electronics", "Airborne Phased Array Radar Module Maintenance",
     "Maintainability design for BIT/FDR/FIR performance requirements and replacement procedures of airborne AESA radar TR modules.",
     4),
    ("electronics", "Data Link Terminal Chassis Maintainability Design",
     "Maintainability requirements for Link-16/22 data link terminal chassis, focusing on LRU granularity, MTTR, and diagnostic coverage rate.",
     3),
    ("electronics", "General-Purpose IFF (Identification Friend or Foe) Interrogator Maintenance",
     "Maintenance scheme for airborne/shipboard IFF transponders, focusing on cryptographic module replacement safety, BIT, and traceable labeling.",
     3),
    ("electronics", "Satellite Communication Terminal Antenna Pointing Mechanism Maintenance",
     "Maintenance design for satellite communication terminal servo pointing mechanisms, focusing on lubrication paths, encoder zero calibration, and preventive maintenance.",
     3),
    ("electronics", "Vehicle-Mounted Power Conversion Module Maintenance",
     "Maintenance scheme for AC-DC/DC-DC power conversion modules, focusing on interchangeability, temperature monitoring, and thermal protection interlocks.",
     3),
]


def main():
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for i, (family, title, desc, n_sugg) in enumerate(SCENARIOS):
            record = {
                "id": f"task3_{i + 1:03d}",
                "query": (
                    f"{title}. {desc} "
                    f"Please provide {n_sugg} structured maintainability design suggestions, each should include"
                    ": ① design key points; ② applicable standard clauses; ③ reference threshold values or quantitative metrics (if any)."
                ),
                "title": title,
                "family": family,
                "scenario_desc": desc,
                "n_suggestions": n_sugg,
                "task": "open_generation",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    from collections import Counter
    ctr = Counter(s[0] for s in SCENARIOS)
    print(f"Wrote {len(SCENARIOS)} scenarios → {OUT_FILE}")
    print("Per-family:", dict(ctr))
    print("N-suggestions range:", {n_sugg for *_, n_sugg in SCENARIOS})


if __name__ == "__main__":
    main()
