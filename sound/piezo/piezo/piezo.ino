#define PIEZO_PIN A0

void setup() {
  Serial.begin(115200);
}

void loop() {
  int valor = analogRead(PIEZO_PIN);
  Serial.println(valor);
  delayMicroseconds(500);
}