int emgExtensorPin = A0;
int emgFlexorPin = A1;

int yPinTop = A4;
int yPinBottom = A5;

unsigned long t;

void setup() {
  Serial.begin(230400);
}

void loop() {
  t = micros();

  int yTop = analogRead(yPinTop);
  int yBottom = analogRead(yPinBottom);
  int emgExtensor = analogRead(emgExtensorPin);
  int emgFlexor = analogRead(emgFlexorPin);

  Serial.print(t);
  Serial.print(",");
  Serial.print(yTop);
  Serial.print(",");
  Serial.print(yBottom);
  Serial.print(",");
  Serial.print(emgExtensor);
  Serial.print(",");
  Serial.println(emgFlexor);
}