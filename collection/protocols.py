"""
Definiciones de bloques y protocolos por sesion.

Vocabulario fijo de tareas (no agregar/renombrar entre sesiones; nuevos
labels solo via nuevas constantes para mantener consistencia historica).
"""
from dataclasses import dataclass, field
from typing import List, Optional


TASK_REST = "rest_baseline"
TASK_SLOW_FLEX = "slow_flex"
TASK_SLOW_EXT = "slow_ext"
TASK_SLOW_CYCLES = "slow_cycles"
TASK_FAST_FLEX = "fast_flex"
TASK_FAST_EXT = "fast_ext"
TASK_HOLDS = "holds_intermediate"
TASK_MICRO = "microcorrections"
TASK_NOISE_FINGERS = "noise_other_fingers"
TASK_NOISE_WRIST = "noise_wrist_grip"
TASK_FREE = "free_controlled"
TASK_TRANSITION = "transition"

TASKS = {
    TASK_REST, TASK_SLOW_FLEX, TASK_SLOW_EXT, TASK_SLOW_CYCLES,
    TASK_FAST_FLEX, TASK_FAST_EXT, TASK_HOLDS, TASK_MICRO,
    TASK_NOISE_FINGERS, TASK_NOISE_WRIST, TASK_FREE, TASK_TRANSITION,
}


@dataclass
class Block:
    task: str
    duration: float
    instructions: str
    cue: Optional[str] = None

    def __post_init__(self):
        if self.task not in TASKS:
            raise ValueError(f"task '{self.task}' no esta en TASKS")
        if self.duration <= 0:
            raise ValueError("duration debe ser > 0")


def base_protocol() -> List[Block]:
    return [
        Block(TASK_REST, 90,
              "Mano relajada sobre la mesa. No contraigas nada. Mira un punto fijo."),
        Block(TASK_SLOW_FLEX, 120,
              "Flexiona el indice MUY lento (~3 s recorrido completo) y solta. Repetir."),
        Block(TASK_SLOW_EXT, 120,
              "Extiende el indice MUY lento (~3 s recorrido completo) y solta. Repetir."),
        Block(TASK_SLOW_CYCLES, 150,
              "Ciclos continuos flex<->ext lentos (~0.25 Hz). Sin pausas.",
              cue="Metronomo a 0.25 Hz si tenes."),
        Block(TASK_FAST_FLEX, 90,
              "Flexion rapida y volve. Pausa ~2 s entre repeticiones."),
        Block(TASK_FAST_EXT, 90,
              "Extension rapida y volve. Pausa ~2 s entre repeticiones."),
        Block(TASK_HOLDS, 150,
              "Mantene 4 s en posiciones 25 %, 50 %, 75 % del rango. Rota entre las tres.",
              cue="Marca visual de target si tenes."),
        Block(TASK_MICRO, 120,
              "Microcorrecciones de 5-10 % del rango alrededor de la mitad."),
        Block(TASK_NOISE_FINGERS, 100,
              "NO muevas el indice. Mueve pulgar, medio, anular, menique uno por vez."),
        Block(TASK_NOISE_WRIST, 100,
              "NO muevas el indice. Flex/ext de muneca, pronacion, agarre con los otros dedos."),
        Block(TASK_FREE, 70,
              "Indice libre, mezcla velocidades y posiciones. Cubri todo el rango."),
    ]


def _protocol_high_distractor() -> List[Block]:
    """Sesiones 7-8: mas negativos."""
    return [
        Block(TASK_REST, 90, "Mano relajada."),
        Block(TASK_SLOW_FLEX, 120, "Flex lenta."),
        Block(TASK_SLOW_EXT, 120, "Ext lenta."),
        Block(TASK_SLOW_CYCLES, 150, "Ciclos lentos."),
        Block(TASK_FAST_FLEX, 90, "Flex rapida."),
        Block(TASK_FAST_EXT, 90, "Ext rapida."),
        Block(TASK_HOLDS, 80, "Holds 25/50/75."),
        Block(TASK_MICRO, 70, "Microcorrecciones."),
        Block(TASK_NOISE_FINGERS, 180,
              "NO indice. Pulgar/medio/anular/menique, alternando velocidad y fuerza."),
        Block(TASK_NOISE_WRIST, 180,
              "NO indice. Muneca, pronacion, agarre, coactivaciones raras."),
        Block(TASK_FREE, 30, "Indice libre breve."),
    ]


def _protocol_test() -> List[Block]:
    """Sesion 10: test puro, NO entrenar.

    Mantiene holds y micro porque, aunque el modelo decodifica peor ahi por
    la ambiguedad del resorte, sigue siendo util medir como se comporta
    el sistema bajo esas condiciones en el test final.
    """
    return [
        Block(TASK_REST, 60, "Mano relajada."),
        Block(TASK_SLOW_CYCLES, 180, "Ciclos lentos continuos."),
        Block(TASK_FAST_FLEX, 90, "Flex rapida + pausas."),
        Block(TASK_FAST_EXT, 90, "Ext rapida + pausas."),
        Block(TASK_HOLDS, 180, "Holds intermedios."),
        Block(TASK_MICRO, 120, "Microcorrecciones."),
        Block(TASK_NOISE_FINGERS, 120, "Distractores otros dedos."),
        Block(TASK_NOISE_WRIST, 120, "Distractores muneca/agarre."),
        Block(TASK_FREE, 240, "Indice libre, todo el rango y velocidades."),
    ]


def protocol_for_session(session_number: int) -> List[Block]:
    if session_number in (7, 8):
        return _protocol_high_distractor()
    if session_number == 10:
        return _protocol_test()
    return base_protocol()


def session_split(session_number: int) -> str:
    """Indica si la sesion va a train o test. Se guarda en metadata."""
    if session_number == 10:
        return "test"
    return "train"
