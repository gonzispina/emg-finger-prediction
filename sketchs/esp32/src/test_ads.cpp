/*
 * ADS1256 standalone diagnostic.
 *
 * Foco: aislar problemas del ADS antes de mezclarlo con joysticks y
 * round-robin. Hace estas cosas en orden y reporta cada paso:
 *
 *   1. Init SPI a frecuencia conservadora (500 kHz).
 *   2. Reset + SDATAC.
 *   3. Lee TODOS los registros pre-config (sanity check de SPI/MISO).
 *   4. Escribe ADCON (buffer ON), DRATE (7500 SPS), MUX (AIN0 vs AINCOM).
 *   5. Re-lee registros post-config para verificar que se escribieron.
 *   6. SELFCAL.
 *   7. Loop: lee 1 sample/cycle de AIN0 + DRDY pin state + dt entre samples.
 *
 * Por que esto y no el firmware completo:
 *   - Un solo canal (no MUX-switching) reduce variables.
 *   - SCLK conservador (500 kHz, no 1.92 MHz) descarta limites de speed.
 *   - Buffer ON (ADCON=0x02) es lo correcto para EMG (alta impedancia).
 *   - Imprime el STATUS register: bits 4-7 deben ser 0x3 (ID del ADS1256).
 *     Si los lees como otra cosa, hay problema de comunicacion / cableado.
 *
 * Como compilar con este source en vez de main.cpp:
 *   pio run -e test_ads --target upload
 *   pio device monitor -e test_ads -b 921600
 */
#include <Arduino.h>
#include <SPI.h>

static const int PIN_SCK   = 38;
static const int PIN_MISO  = 41;
static const int PIN_MOSI  = 1;
static const int PIN_CS    = 2;
static const int PIN_DRDY  = 4;

static const uint32_t SPI_HZ = 500000;     // 500 kHz, conservador

// Commands
static const uint8_t CMD_WAKEUP  = 0x00;
static const uint8_t CMD_RDATA   = 0x01;
static const uint8_t CMD_SDATAC  = 0x0F;
static const uint8_t CMD_SYNC    = 0xFC;
static const uint8_t CMD_RESET   = 0xFE;
static const uint8_t CMD_SELFCAL = 0xF0;

// Register addresses
static const uint8_t REG_STATUS = 0x00;
static const uint8_t REG_MUX    = 0x01;
static const uint8_t REG_ADCON  = 0x02;
static const uint8_t REG_DRATE  = 0x03;
static const uint8_t REG_IO     = 0x04;

SPIClass spiADS(HSPI);

inline void csLow()  { digitalWrite(PIN_CS, LOW); }
inline void csHigh() { digitalWrite(PIN_CS, HIGH); }

void adsCommand(uint8_t cmd) {
  csLow();
  delayMicroseconds(2);
  spiADS.transfer(cmd);
  delayMicroseconds(2);
  csHigh();
}

void adsWriteReg(uint8_t reg, uint8_t val) {
  csLow();
  delayMicroseconds(2);
  spiADS.transfer(0x50 | reg);
  spiADS.transfer(0x00);
  spiADS.transfer(val);
  delayMicroseconds(2);
  csHigh();
}

uint8_t adsReadReg(uint8_t reg) {
  csLow();
  delayMicroseconds(2);
  spiADS.transfer(0x10 | reg);
  spiADS.transfer(0x00);
  delayMicroseconds(10);
  uint8_t v = spiADS.transfer(0xFF);
  delayMicroseconds(2);
  csHigh();
  return v;
}

bool waitDrdy(uint32_t timeout_us = 200000) {
  uint32_t t0 = micros();
  while (digitalRead(PIN_DRDY)) {
    if (micros() - t0 > timeout_us) return false;
  }
  return true;
}

int32_t adsReadData24() {
  csLow();
  delayMicroseconds(2);
  spiADS.transfer(CMD_RDATA);
  delayMicroseconds(10);
  uint8_t b2 = spiADS.transfer(0xFF);
  uint8_t b1 = spiADS.transfer(0xFF);
  uint8_t b0 = spiADS.transfer(0xFF);
  delayMicroseconds(2);
  csHigh();
  int32_t v = ((int32_t)b2 << 16) | ((int32_t)b1 << 8) | b0;
  if (v & 0x800000) v |= 0xFF000000;
  return v;
}

void printAllRegs(const char* label) {
  Serial.print("--- Registers ");
  Serial.print(label);
  Serial.println(" ---");
  uint8_t status = adsReadReg(REG_STATUS);
  Serial.printf("  STATUS = 0x%02X   (ID bits 4-7 = 0x%X, debe ser 0x3 en ADS1256)\n",
                status, (status >> 4) & 0x0F);
  Serial.printf("  MUX    = 0x%02X\n", adsReadReg(REG_MUX));
  Serial.printf("  ADCON  = 0x%02X\n", adsReadReg(REG_ADCON));
  Serial.printf("  DRATE  = 0x%02X\n", adsReadReg(REG_DRATE));
  Serial.printf("  IO     = 0x%02X\n", adsReadReg(REG_IO));
  Serial.printf("  DRDY pin (HIGH=esperando, LOW=listo) = %d\n",
                digitalRead(PIN_DRDY));
}

void setup() {
  Serial.begin(921600);
  delay(2000);
  Serial.println();
  Serial.println("=================================================");
  Serial.println("  ADS1256 STANDALONE DIAGNOSTIC");
  Serial.println("=================================================");
  Serial.printf("  SCLK=%d MISO=%d MOSI=%d CS=%d DRDY=%d\n",
                PIN_SCK, PIN_MISO, PIN_MOSI, PIN_CS, PIN_DRDY);
  Serial.printf("  SPI HZ = %u (conservador)\n", SPI_HZ);
  Serial.println();

  pinMode(PIN_CS, OUTPUT);
  pinMode(PIN_DRDY, INPUT_PULLUP);
  csHigh();

  spiADS.begin(PIN_SCK, PIN_MISO, PIN_MOSI, PIN_CS);
  spiADS.beginTransaction(SPISettings(SPI_HZ, MSBFIRST, SPI_MODE1));
  delay(50);

  Serial.println("[1] RESET");
  adsCommand(CMD_RESET);
  delay(20);

  Serial.println("[2] SDATAC");
  adsCommand(CMD_SDATAC);
  delayMicroseconds(50);

  printAllRegs("DESPUES DE RESET");
  Serial.println();
  Serial.println("Valores esperados post-reset:");
  Serial.println("  STATUS = 0x3X (X depende de BUFEN/etc, alto nibble 0x3 = ID)");
  Serial.println("  MUX    = 0x01 (default: AIN0+ AIN1-)");
  Serial.println("  ADCON  = 0x20 (default)");
  Serial.println("  DRATE  = 0xF0 (default: 30000 SPS)");
  Serial.println();

  Serial.println("[3] WRITE config:");
  Serial.println("    STATUS = 0x00 (BUFEN=off, ACAL=off, ORDER=MSB)");
  Serial.println("    ADCON  = 0x00 (gain=1, CLKOUT off, SDCS off)");
  Serial.println("    DRATE  = 0xD0 (7500 SPS)");
  Serial.println("    MUX    = 0x01 (AIN0(+) AIN1(-) diferencial)");
  adsWriteReg(REG_STATUS, 0x00);
  adsWriteReg(REG_ADCON,  0x00);
  adsWriteReg(REG_DRATE,  0xD0);
  adsWriteReg(REG_MUX,    0x01);
  delay(5);

  printAllRegs("DESPUES DE WRITE");
  Serial.println();

  Serial.println("[4] SELFCAL");
  adsCommand(CMD_SELFCAL);
  bool ok = waitDrdy(500000);
  Serial.printf("    SELFCAL completo: %s\n", ok ? "OK" : "TIMEOUT");
  delay(10);

  Serial.println("[5] SYNC + WAKEUP para arrancar conversiones");
  adsCommand(CMD_SYNC);
  delayMicroseconds(5);
  adsCommand(CMD_WAKEUP);

  Serial.println();
  Serial.println("[6] Lectura rotando entre 4 pares diferenciales:");
  Serial.println("    AIN0-AIN1  AIN2-AIN3  AIN4-AIN5  AIN6-AIN7");
  Serial.println("    Cableado fisico de cada EMG pill:");
  Serial.println("      OUT -> AIN par (0, 2, 4, 6)");
  Serial.println("      GND -> AIN impar (1, 3, 5, 7)");
  Serial.println();
  Serial.println("    Formato: ch=NAME  raw_hex  dec  V");
  Serial.println();
}

// Definicion de canales a probar (4 pares diferenciales)
struct ChannelTest {
  const char* name;
  uint8_t mux;
};
static const ChannelTest CHANNELS[] = {
  {"AIN0(+)-AIN1(-) ext_ind", 0x01},
  {"AIN2(+)-AIN3(-) flx_ind", 0x23},
  {"AIN4(+)-AIN5(-) ext_med", 0x45},
  {"AIN6(+)-AIN7(-) flx_med", 0x67},
};
static const int N_CHAN = sizeof(CHANNELS) / sizeof(CHANNELS[0]);

void loop() {
  static int idx = 0;
  static uint32_t reads = 0;

  // 1) Configurar el canal actual
  adsWriteReg(REG_MUX, CHANNELS[idx].mux);
  adsCommand(CMD_SYNC);
  delayMicroseconds(5);
  adsCommand(CMD_WAKEUP);

  // 2) Esperar DRDY (tras MUX switch hay settling time = 1/SPS = 133us @ 7500SPS)
  if (!waitDrdy(500000)) {
    Serial.printf("  ! DRDY timeout en %s\n", CHANNELS[idx].name);
    delay(200);
    idx = (idx + 1) % N_CHAN;
    return;
  }

  // 3) Leer y mostrar
  int32_t raw = adsReadData24();
  reads++;
  float volts = (float)raw * 2.5f / 8388608.0f;

  Serial.printf("  ch=%-22s  raw=0x%06lX  dec=%+9ld  V=%+9.6f  (#%u)\n",
                CHANNELS[idx].name,
                (long)(raw & 0xFFFFFF), (long)raw, volts, reads);

  idx = (idx + 1) % N_CHAN;
  delay(80);   // un poco mas lento para que se vea claro en la consola
}
