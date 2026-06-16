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
static const uint32_t SPI_HZ = 100000;

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
  delayMicroseconds(10);
}

void adsWriteReg(uint8_t reg, uint8_t val) {
  csLow();
  spiADS.transfer(0x50 | reg);
  spiADS.transfer(0x00);
  spiADS.transfer(val);
  csHigh();
  delayMicroseconds(10);
}

uint8_t adsReadReg(uint8_t reg) {
  csLow();
  spiADS.transfer(0x10 | reg);
  spiADS.transfer(0x00);
  delayMicroseconds(10);
  uint8_t v = spiADS.transfer(0xFF);
  csHigh();
  delayMicroseconds(10);
  return v;
}

bool waitDrdy(uint32_t timeout_us = 10000) {
  uint32_t t0 = micros();
  while (digitalRead(PIN_DRDY) == HIGH) {
    if ((uint32_t)(micros() - t0) > timeout_us) return false;
  }
  return true;
}

void setSingleEndedAIN0() {
  // AIN0 - AINCOM/GND
  adsWriteReg(REG_MUX, 0x08);
  adsCommand(CMD_SYNC);
  delayMicroseconds(10);
  adsCommand(CMD_WAKEUP);
}

int32_t adsReadData24() {
  if (!waitDrdy()) return INT32_MIN;

  csLow();
  spiADS.transfer(CMD_RDATA);
  delayMicroseconds(10);

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

void setup() {
  Serial.begin(921600);
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
  // DRATE  0xA1 = 1000 SPS
  adsWriteReg(REG_STATUS, 0x00);
  adsWriteReg(REG_ADCON,  0x00);
  adsWriteReg(REG_DRATE,  0xA1);

  adsCommand(CMD_SELFCAL);
  waitDrdy(50000);
  delay(10);

  setSingleEndedAIN0();

  analogReadResolution(12);

  Serial.println("# SINGLE-ENDED TEST");
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
  emg1 = 0;
  emg2 = 0;
  emg3 = 0;

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

  Serial.write(frame, FRAME_SIZE);
}