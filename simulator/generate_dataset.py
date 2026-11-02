
# ==========================================================
# ACI606 - CLOUD FLEET CHALLENGE
# Generador de dataset para Machine Learning
# ==========================================================

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from core import make_robot, step_robot


# ==========================================================
# 1. CONFIGURACION
# ==========================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="Generate fleet training dataset"
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=3000
    )

    parser.add_argument(
        "--robots",
        type=int,
        default=8
    )

    parser.add_argument(
        "--fleets",
        type=int,
        default=20
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=606
    )

    parser.add_argument(
        "--output",
        default="data/training.csv"
    )

    return parser.parse_args()


# ==========================================================
# 2. GENERACION DE DATOS
# ==========================================================

def main() -> None:

    args = parse_args()

    # Validar parametros
    if args.samples <= 0:
        raise SystemExit("Samples debe ser mayor que cero.")

    if args.robots <= 0:
        raise SystemExit("Robots debe ser mayor que cero.")

    if args.fleets < 2:
        raise SystemExit("Se necesitan al menos dos flotas.")

    if args.samples < args.fleets:
        raise SystemExit(
            "Samples debe ser mayor o igual que fleets."
        )

    rng = random.Random(args.seed)

    rows = []

    # Distribuir registros entre las flotas
    base, remainder = divmod(
        args.samples,
        args.fleets
    )

    # Generar cada flota de manera independiente
    for fleet_index in range(args.fleets):

        fleet_id = f"F{fleet_index + 1:03d}"

        # Crear nuevos robots para cada flota
        robots = [
            make_robot(i + 1, rng)
            for i in range(args.robots)
        ]

        fleet_samples = base + (
            1 if fleet_index < remainder else 0
        )

        for i in range(fleet_samples):

            robot = robots[i % len(robots)]

            sample, status = step_robot(
                robot,
                rng
            )

            # Incorporar identificador de flota
            rows.append({
                **sample,
                "fleet_id": fleet_id,
                "status": status
            })

    # ======================================================
    # 3. GUARDAR CSV
    # ======================================================

    output = Path(args.output)

    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fieldnames = list(rows[0].keys())

    with output.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)

    # ======================================================
    # 4. RESUMEN DEL DATASET
    # ======================================================

    classes = [
        "NORMAL",
        "WARNING",
        "CRITICAL"
    ]

    counts = {
        label: sum(
            row["status"] == label
            for row in rows
        )
        for label in classes
    }

    print("\n========== DATASET GENERADO ==========")

    print("Archivo:", output)
    print("Registros:", len(rows))
    print("Flotas:", args.fleets)
    print("Robots por flota:", args.robots)
    print("Semilla:", args.seed)

    print("\nDistribucion de clases:")

    for label, count in counts.items():

        percentage = count / len(rows) * 100

        print(
            f"{label}: {count} ({percentage:.2f}%)"
        )

    print("\nGeneracion finalizada correctamente.")


if __name__ == "__main__":
    main()
