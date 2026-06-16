---
name: emg-data-collection
description: Guia al usuario para correr una sesion de captura de EMG/finger para el proyecto robothand. Usar cuando el usuario quiera grabar datos, "tomar muestras", "correr una sesion", o cuando mencione subject_id/session_number/placement_id en el contexto de este proyecto.
---

# EMG / Finger data collection — robothand

Sistema de captura para entrenar una TCN causal que predice velocidad/delta del dedo indice a partir de 2 canales EMG (flexor + extensor). Hardware via Serial.

## Punto de entrada

```bash
python collect.py                                  # prompts interactivos
python collect.py -s P01 -n 1 -p P01-A             # bypass de prompts
```

Defaults: `--port /dev/cu.usbserial-2110`, `--baud 230400`, `--output-dir data/raw`.

## Estructura

- `collect.py` — entry CLI
- `collection/protocols.py` — `Block`, vocabulario fijo de tareas (`TASKS`), `protocol_for_session(n)`, `session_split(n)`
- `collection/serial_io.py` — `SerialReader`, `Sample`, `parse_5field` (default, matchea el sketch en `sketchs/arduino/`)
- `collection/runner.py` — `run_session()`, `CSV_COLUMNS`
- `data/raw/<subject_id>/<session_id>.csv` + `.json`

`session_id = "{subject}_S{NN}_{YYYYMMDD_HHMMSS}"`.

## Protocolo (20 min, 11 bloques) — sesiones 1-6, 9

| # | task                  | s   |
|---|-----------------------|-----|
| 1 | rest_baseline         | 90  |
| 2 | slow_flex             | 120 |
| 3 | slow_ext              | 120 |
| 4 | slow_cycles           | 150 |
| 5 | fast_flex             | 90  |
| 6 | fast_ext              | 90  |
| 7 | holds_intermediate    | 150 |
| 8 | microcorrections      | 120 |
| 9 | noise_other_fingers   | 100 |
| 10| noise_wrist_grip      | 100 |
| 11| free_controlled       | 70  |

**Variantes:**
- Sesiones 1-4: placement consistente (`<subject>-A`).
- Sesiones 5-6: mismo placement, leve cambio (`-B`).
- Sesiones 7-8: `_protocol_high_distractor` (noise_* extendidos a 180s).
- Sesion 9: placement bastante diferente (`-C`).
- Sesion 10: `_protocol_test`, `split="test"`, **NO entrenar**.

`task="transition"` se usa para los 3 s de countdown entre bloques — filtrar antes de entrenar.

## CSV schema

`pc_timestamp, t_micros, y_top, y_bottom, flexor, extensor, trial_id, task, session_id, subject_id, placement_id, optional_notes`

Serial = `t_micros, y_top, y_bottom, emg_extensor, emg_flexor` (5 ints, formato del sketch actual). `y_top` e `y_bottom` son dos joysticks verticales con el dedo entre ellos; la posicion del dedo es `y_top + y_bottom` con gaussian smoothing 1D, calculada offline en el pipeline de preprocesamiento.

## Cuando guiar una sesion

1. Confirmar `subject_id`, `session_number`, `placement_id`, port, baud.
2. Recordar al usuario las precondiciones (ver checklist abajo) ANTES de correr.
3. Lanzar `python collect.py ...`.
4. No interpretar datos en vivo — solo recoleccion cruda.

## Hardware actual

- **MCU: Arduino Uno** (planeada migracion a ESP32). Sampling efectivo observado en sesion ~605 Hz, no 800. La fs real se estima desde `t_micros` en el pipeline offline.
- Si en algun momento el usuario cambia el MCU, anotar en `optional_notes` (e.g. `"mcu=esp32"`) y considerar marcar como nuevo `placement_id` o serie distinta. Los datos pre/post migracion no son intercambiables.

## Checklist pre-sesion (recordar al usuario)

- Sujeto sentado, antebrazo apoyado, en la misma postura que sesiones previas.
- Electrodos: foto + verificar landmarks (distancia desde epicondilo medial, eje del antebrazo). El `placement_id` debe matchear lo que esta en la hoja de registro.
- Hardware encendido y enchufado al puerto correcto (`ls /dev/cu.*` para chequear). Confirmar que sigue siendo Arduino Uno.
- Sin moverse durante los countdowns.
- Si la sesion previa fue hace < 30 min, posible fatiga — confirmar.

## Memoria viva (mantener actualizado)

- Vocabulario `TASKS` es cerrado. Si hace falta un task nuevo, agregar constante; **nunca renombrar uno existente** (rompe sesiones viejas).
- Si cambia el formato serial, agregar nuevo `parse_*` en `serial_io.py`, no modificar los existentes.
- Las recomendaciones de consistencia entre sesiones estan en la conversacion original — pedirselas al usuario si las necesita.
