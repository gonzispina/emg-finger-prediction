#include <Arduino.h>
#include <SPI.h>

#define FRAME_SIZE 27
#define MAGIC_HI 0xAA
#define MAGIC_LO 0x55

// ADS1256
static const int PIN_SCK  = 38;
static const int PIN_MISO = 41; // ADS DOUT
static const int PIN_MOSI = 1;  // ADS DIN
static const int PIN_CS   = 2;
static const int PIN_DRDY = 4;

// Joysticks
static const int PIN_JOY_A = 5;
static const int PIN_JOY_B = 6;

SPIClass spiADS(HSPI);
static const uint32_t SPI_HZ = 1920000;   // 1.92 MHz (max spec del ADS1256 = fclkin/4)

static const uint8_t CMD_WAKEUP  = 0x00;
static const uint8_t CMD_RDATA   = 0x01;
static const uint8_t CMD_SDATAC  = 0x0F;
static const uint8_t CMD_SYNC    = 0xFC;
static const uint8_t CMD_RESET   = 0xFE;
static const uint8_t CMD_SELFCAL = 0xF0;

static const uint8_t REG_STATUS = 0x00;
static const uint8_t REG_MUX    = 0x01;
static const uint8_t REG_ADCON  = 0x02;
static const uint8_t REG_DRATE  = 0x03;

inline void csLow()  { digitalWrite(PIN_CS, LOW); }
inline void csHigh() { digitalWrite(PIN_CS, HIGH); }

void adsCommand(uint8_t cmd) {
  csLow();
  spiADS.transfer(cmd);
  csHigh();
  // sin delay post-CS — el siguiente CS LOW da margen de por si
}

void adsWriteReg(uint8_t reg, uint8_t val) {
  csLow();
  spiADS.transfer(0x50 | reg);
  spiADS.transfer(0x00);
  spiADS.transfer(val);
  csHigh();
}

uint8_t adsReadReg(uint8_t reg) {
  csLow();
  spiADS.transfer(0x10 | reg);
  spiADS.transfer(0x00);
  delayMicroseconds(10);            // t6 spec antes del read
  uint8_t v = spiADS.transfer(0xFF);
  csHigh();
  return v;
}

bool waitDrdy(uint32_t timeout_us = 10000) {
  uint32_t t0 = micros();
  while (digitalRead(PIN_DRDY) == HIGH) {
    if ((uint32_t)(micros() - t0) > timeout_us) return false;
  }
  return true;
}

// Switch al canal y gain indicados. ADCON: PGA gain bits 2:0.
//   gain 1 = 0x00 (00 = no clock out, 00 = no sensor detect, 000 = PGA 1)
//   gain 2 = 0x01,  gain 4 = 0x02,  gain 8 = 0x03,
//   gain 16 = 0x04, gain 32 = 0x05, gain 64 = 0x06.
// delay 4us = t11 spec (24 t_CLKIN).
void switchChannel(uint8_t muxValue, uint8_t adconValue) {
  adsWriteReg(REG_ADCON, adconValue);
  adsWriteReg(REG_MUX, muxValue);
  adsCommand(CMD_SYNC);
  delayMicroseconds(4);
  adsCommand(CMD_WAKEUP);
}
// Todos los canales con gain=1 (ADCON=0x00).
void setSingleEndedAIN0() { switchChannel(0x08, 0x00); }   // ext_ind
void setSingleEndedAIN2() { switchChannel(0x28, 0x00); }   // flx_ind
void setSingleEndedAIN4() { switchChannel(0x48, 0x00); }   // ext_med
void setSingleEndedAIN6() { switchChannel(0x68, 0x00); }   // flx_med

int32_t adsReadData24() {
  if (!waitDrdy()) return INT32_MIN;

  csLow();
  spiADS.transfer(CMD_RDATA);
  delayMicroseconds(7);                 // t6 spec (50 t_CLKIN @ 7.68 MHz)

  uint8_t b2 = spiADS.transfer(0xFF);
  uint8_t b1 = spiADS.transfer(0xFF);
  uint8_t b0 = spiADS.transfer(0xFF);

  csHigh();

  int32_t v = ((int32_t)b2 << 16) | ((int32_t)b1 << 8) | b0;
  if (v & 0x800000) v |= 0xFF000000;
  return v;
}

uint8_t crc8(const uint8_t* data, size_t len) {
  uint8_t crc = 0;
  for (size_t i = 0; i < len; i++) {
    crc ^= data[i];
    for (uint8_t b = 0; b < 8; b++) {
      crc = (crc & 0x80) ? (uint8_t)((crc << 1) ^ 0x07) : (uint8_t)(crc << 1);
    }
  }
  return crc;
}

uint8_t frame[FRAME_SIZE];

// ---------- Dual-core: serial output corre en core 0 ----------
// Core 0: serialTask consume la queue y hace Serial.write
// Core 1 (default Arduino loop): lee ADS + joys + push a queue
// Esto desacopla el costo de Serial.write del loop de muestreo.
static QueueHandle_t g_frame_queue;
static const int QUEUE_CAPACITY = 128;     // ~64ms de buffering a 2kHz

void serialTask(void* params) {
  uint8_t buf[FRAME_SIZE];
  for (;;) {
    if (xQueueReceive(g_frame_queue, buf, portMAX_DELAY) == pdTRUE) {
      Serial.write(buf, FRAME_SIZE);
    }
  }
}

void setup() {
  Serial.begin(2000000);    // 2 Mbps (USB-CDC: baudrate virtual, no UART real)
  delay(2000);

  pinMode(PIN_CS, OUTPUT);
  pinMode(PIN_DRDY, INPUT_PULLUP);
  csHigh();

  spiADS.begin(PIN_SCK, PIN_MISO, PIN_MOSI, PIN_CS);
  spiADS.beginTransaction(SPISettings(SPI_HZ, MSBFIRST, SPI_MODE1));

  adsCommand(CMD_RESET);
  delay(100);

  adsCommand(CMD_SDATAC);
  delay(10);

  // STATUS 0x00 = buffer OFF
  // ADCON  0x00 = gain 1
  // DRATE  0xE0 = 15000 SPS (settling ~67us, mejor SNR que 30 kSPS)
  adsWriteReg(REG_STATUS, 0x00);
  adsWriteReg(REG_ADCON,  0x00);
  adsWriteReg(REG_DRATE,  0xE0);

  adsCommand(CMD_SELFCAL);
  waitDrdy(50000);
  delay(10);

  setSingleEndedAIN0();

  analogReadResolution(12);

  Serial.println("# 4-CH SINGLE-ENDED: emg0=AIN0 emg1=AIN2 emg2=AIN4 emg3=AIN6 (vs AINCOM)");
  Serial.println("# Gains: emg0=1 emg1=1 emg2=1 emg3=1");
  Serial.println("# Dual-core: ADC en core 1, Serial.write en core 0");
  Serial.flush();

  // Crear queue de frames y arrancar el task de serial en core 0.
  g_frame_queue = xQueueCreate(QUEUE_CAPACITY, FRAME_SIZE);
  xTaskCreatePinnedToCore(
    serialTask, "serial", 4096, NULL,
    1,           // prioridad media-baja (no compite con loop)
    NULL,
    0            // core 0
  );
  Serial.print("# STATUS=0x"); Serial.println(adsReadReg(REG_STATUS), HEX);
  Serial.print("# MUX=0x");    Serial.println(adsReadReg(REG_MUX), HEX);
  Serial.print("# ADCON=0x");  Serial.println(adsReadReg(REG_ADCON), HEX);
  Serial.print("# DRATE=0x");  Serial.println(adsReadReg(REG_DRATE), HEX);
  Serial.println("# frame: AA55 t_us[4] emg[4] joy_a[2] joy_b[2] crc[1]");
  Serial.flush();
}

void loop() {
  int32_t emg0, emg1, emg2, emg3;

  setSingleEndedAIN0();
  emg0 = adsReadData24();

  setSingleEndedAIN2();
  emg1 = adsReadData24();

  setSingleEndedAIN4();
  emg2 = adsReadData24();

  setSingleEndedAIN6();
  emg3 = adsReadData24();

  uint16_t joy_a = (uint16_t)analogReadRaw(PIN_JOY_A);
  uint16_t joy_b = (uint16_t)analogReadRaw(PIN_JOY_B);

  uint32_t t = micros();

  frame[0] = MAGIC_HI;
  frame[1] = MAGIC_LO;
  memcpy(&frame[2],  &t,     4);
  memcpy(&frame[6],  &emg0,  4);
  memcpy(&frame[10], &emg1,  4);
  memcpy(&frame[14], &emg2,  4);
  memcpy(&frame[18], &emg3,  4);
  memcpy(&frame[22], &joy_a, 2);
  memcpy(&frame[24], &joy_b, 2);
  frame[26] = crc8(&frame[2], 24);

  // Push al queue para que el task de core 0 haga el Serial.write.
  // Si la queue esta llena, descartamos el frame (mejor que bloquear el
  // loop de muestreo). Timeout 0 = no block.
  xQueueSend(g_frame_queue, frame, 0);
}