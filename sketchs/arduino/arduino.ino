// Frame binario de 31 bytes — mismo formato que el ESP32 para compatibilidad:
//   [0..1]   magic 0xAA 0x55
//   [2..5]   t_micros        uint32 LE
//   [6..9]   emg0 (ext_ind)  int32  LE   <- analogRead(A0), cast a int32
//   [10..13] emg1 (flx_ind)  int32  LE   <- analogRead(A1)
//   [14..17] emg2            int32  LE   = 0  (no hay canal medio)
//   [18..21] emg3            int32  LE   = 0
//   [22..23] joy_ext_ind     uint16 LE   <- analogRead(A2)
//   [24..25] joy_flex_ind    uint16 LE   <- analogRead(A3)
//   [26..27] joy_ext_med     uint16 LE   = 0
//   [28..29] joy_flex_med    uint16 LE   = 0
//   [30]     CRC-8 (poly 0x07, init 0x00) sobre bytes [2..29]
//
// Baud: 1000000 (exacto en AVR 16 MHz — 0% error de redondeo).
// Para conectar desde Python: --baud 1000000

static const int PIN_EMG_EXT = A0;   // emg0 extensor indice
static const int PIN_EMG_FLX = A1;   // emg1 flexor   indice
static const int PIN_JOY_EXT = A2;   // joy_ext_ind
static const int PIN_JOY_FLX = A3;   // joy_flex_ind

static const uint8_t FRAME_SIZE = 31;
static uint8_t frame[FRAME_SIZE];

static uint8_t crc8(const uint8_t* data, uint8_t len) {
  uint8_t crc = 0;
  for (uint8_t i = 0; i < len; i++) {
    crc ^= data[i];
    for (uint8_t b = 0; b < 8; b++) {
      crc = (crc & 0x80) ? (uint8_t)((crc << 1) ^ 0x07) : (uint8_t)(crc << 1);
    }
  }
  return crc;
}

// Target fs del modelo de entrenamiento
static const uint32_t PERIOD_US = 1650;  // ~606 Hz (1e6 / 606)
static uint32_t next_us = 0;

void setup() {
  Serial.begin(1000000);
  next_us = micros();
}

void loop() {
  // Espera hasta el proximo tick sin bloquear con delay()
  while ((int32_t)(micros() - next_us) < 0) {}
  next_us += PERIOD_US;

  int32_t  emg0         = (int32_t)analogRead(PIN_EMG_EXT);
  int32_t  emg1         = (int32_t)analogRead(PIN_EMG_FLX);
  int32_t  emg2         = 0;
  int32_t  emg3         = 0;
  uint16_t joy_ext_ind  = (uint16_t)analogRead(PIN_JOY_EXT);
  uint16_t joy_flex_ind = (uint16_t)analogRead(PIN_JOY_FLX);
  uint16_t joy_ext_med  = 0;
  uint16_t joy_flex_med = 0;
  uint32_t t            = micros();

  frame[0] = 0xAA;
  frame[1] = 0x55;
  memcpy(&frame[2],  &t,            4);
  memcpy(&frame[6],  &emg0,         4);
  memcpy(&frame[10], &emg1,         4);
  memcpy(&frame[14], &emg2,         4);
  memcpy(&frame[18], &emg3,         4);
  memcpy(&frame[22], &joy_ext_ind,  2);
  memcpy(&frame[24], &joy_flex_ind, 2);
  memcpy(&frame[26], &joy_ext_med,  2);
  memcpy(&frame[28], &joy_flex_med, 2);
  frame[30] = crc8(&frame[2], 28);

  Serial.write(frame, FRAME_SIZE);
}
