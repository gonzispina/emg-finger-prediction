"""
Runner de sesion: guia por consola, lee serial, escribe CSV + JSON.
"""
import csv
import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .protocols import Block, TASK_TRANSITION, protocol_for_session, session_split
from .serial_io import SerialReader


CSV_COLUMNS = [
    "pc_timestamp",
    "t_micros",
    "emg0",            # extensor del indice (ADS AIN0)
    "emg1",            # flexor   del indice (ADS AIN2)
    "emg2",            # extensor del medio  (ADS AIN4)
    "emg3",            # flexor   del medio  (ADS AIN6)
    "joy_ext_ind",     # eje extensor del joystick del indice (ESP32 GPIO5)
    "joy_flex_ind",    # eje flexor   del joystick del indice (ESP32 GPIO6)
    "joy_ext_med",     # eje extensor del joystick del medio  (ESP32 GPIO7)
    "joy_flex_med",    # eje flexor   del joystick del medio  (ESP32 GPIO8)
    "trial_id",
    "task",
    "session_id",
    "subject_id",
    "placement_id",
    "optional_notes",
]


def make_session_id(subject_id: str, session_number: int) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{subject_id}_S{session_number:02d}_{ts}"


def _record_phase(
    reader: SerialReader,
    writer,
    duration: float,
    task: str,
    trial_id: int,
    session_id: str,
    subject_id: str,
    placement_id: str,
    notes: str,
    show_countdown: bool,
) -> int:
    start = time.time()
    next_tick = start + 1.0
    written = 0
    while True:
        now = time.time()
        elapsed = now - start
        if elapsed >= duration:
            break
        if show_countdown and now >= next_tick:
            remaining = max(0, int(round(duration - elapsed)))
            print(f"   ...{remaining}", flush=True)
            next_tick += 1.0
        s = reader.read()
        if s is None:
            continue
        writer.writerow([
            time.time(),
            s.t_micros,
            s.emg0,
            s.emg1,
            s.emg2,
            s.emg3,
            s.joy_ext_ind,
            s.joy_flex_ind,
            s.joy_ext_med,
            s.joy_flex_med,
            trial_id,
            task,
            session_id,
            subject_id,
            placement_id,
            notes,
        ])
        written += 1
    return written


def run_session(
    subject_id: str,
    session_number: int,
    placement_id: str,
    port: str,
    baud: int = 2000000,
    output_dir: Path = Path("data/raw"),
    notes: str = "",
    countdown_seconds: int = 3,
    blocks_override: Optional[List[Block]] = None,
) -> Path:
    blocks: List[Block] = (blocks_override
                           if blocks_override is not None
                           else protocol_for_session(session_number))
    session_id = make_session_id(subject_id, session_number)
    split = session_split(session_number)

    subject_dir = Path(output_dir) / subject_id
    subject_dir.mkdir(parents=True, exist_ok=True)
    csv_path = subject_dir / f"{session_id}.csv"
    meta_path = subject_dir / f"{session_id}.json"

    metadata = {
        "session_id": session_id,
        "subject_id": subject_id,
        "session_number": session_number,
        "placement_id": placement_id,
        "split": split,
        "port": port,
        "baud": baud,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "csv": csv_path.name,
        "csv_columns": CSV_COLUMNS,
        "serial_format": "binary_v2_31B_AA55_t_4xemg_4xjoy_crc8",
        "countdown_seconds": countdown_seconds,
        "notes": notes,
        "blocks": [
            {"trial_id": i + 1, **asdict(b)} for i, b in enumerate(blocks)
        ],
        "total_task_seconds": sum(b.duration for b in blocks),
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n== SESSION {session_id} ==")
    print(f"  subject     = {subject_id}")
    print(f"  session     = {session_number}  (split={split})")
    print(f"  placement   = {placement_id}")
    print(f"  port/baud   = {port} @ {baud}")
    print(f"  csv         -> {csv_path}")
    print(f"  meta        -> {meta_path}")
    print(f"  bloques     = {len(blocks)}  total ~ {metadata['total_task_seconds']:.0f}s")
    if split == "test":
        print("  ATENCION: esta sesion es TEST. No usar para entrenar.")
    if notes:
        print(f"  notes       = {notes}")
    print()

    reader = SerialReader(port, baud)
    time.sleep(2)
    reader.flush_input()

    print("Preparate. Comienza en:")
    for n in range(5, 0, -1):
        print(f"  {n}...", flush=True)
        time.sleep(1)

    try:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_COLUMNS)

            for trial_id, block in enumerate(blocks, start=1):
                print(f"\n[{trial_id:>2}/{len(blocks)}] {block.task.upper()}  ({block.duration:.0f}s)")
                print(f"   {block.instructions}")
                if block.cue:
                    print(f"   cue: {block.cue}")

                # Transicion (countdown) — etiquetada como 'transition'
                _record_phase(
                    reader, writer,
                    duration=float(countdown_seconds),
                    task=TASK_TRANSITION,
                    trial_id=trial_id,
                    session_id=session_id,
                    subject_id=subject_id,
                    placement_id=placement_id,
                    notes=notes,
                    show_countdown=True,
                )

                # Bloque real
                written = _record_phase(
                    reader, writer,
                    duration=block.duration,
                    task=block.task,
                    trial_id=trial_id,
                    session_id=session_id,
                    subject_id=subject_id,
                    placement_id=placement_id,
                    notes=notes,
                    show_countdown=False,
                )
                print(f"   ok ({written} muestras)")
                f.flush()

            print("\nProtocolo terminado.")

    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario.")

    finally:
        reader.close()
        metadata["ended_at"] = datetime.now().isoformat(timespec="seconds")
        metadata["good_lines"] = reader.good_lines
        metadata["bad_lines"] = reader.bad_lines
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"Lineas validas: {reader.good_lines}  corruptas: {reader.bad_lines}")

    return csv_path
