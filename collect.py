"""
Entry point para correr una sesion de captura.

Uso:
    python collect.py                       # pregunta todo por consola
    python collect.py -s P01 -n 1 -p P01-A  # bypass de prompts
"""
import argparse
from pathlib import Path

from collection.runner import run_session


def _prompt(label: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    val = input(f"{label}{suffix}: ").strip()
    return val or default


def main():
    p = argparse.ArgumentParser(description="Captura EMG/finger para TCN")
    p.add_argument("--subject", "-s", help="subject_id, ej P01")
    p.add_argument("--session", "-n", type=int, help="session_number (1-10)")
    p.add_argument("--placement", "-p", help="placement_id, ej P01-A")
    p.add_argument("--port", default="/dev/cu.usbserial-2110")
    p.add_argument("--baud", type=int, default=2000000)
    p.add_argument("--output-dir", default="data/raw", type=Path)
    p.add_argument("--notes", default="")
    args = p.parse_args()

    subject_id = args.subject or _prompt("subject_id (ej P01)")
    if not subject_id:
        raise SystemExit("subject_id es requerido")

    session_number = args.session
    if session_number is None:
        session_number = int(_prompt("session_number (1-10)", "1"))

    placement_id = args.placement or _prompt(
        "placement_id (ej P01-A)", f"{subject_id}-A"
    )
    notes = args.notes or _prompt("optional_notes (Enter para vacio)")

    run_session(
        subject_id=subject_id,
        session_number=session_number,
        placement_id=placement_id,
        port=args.port,
        baud=args.baud,
        output_dir=args.output_dir,
        notes=notes,
    )


if __name__ == "__main__":
    main()
