
float t;
unsigned long ti, dt = 50, t0, t1; // dt = 50mS, dt = t1 - t0

// Motor R
int interruptPinA = 2;
int countA = 0;

// Motor L
int interruptPinB = 3;
int countB = 0;

//
void Interrupt_A () {
  countA++;
}

void Interrupt_B () {
  countB++;
}

//
void setup() {
  attachInterrupt(digitalPinToInterrupt(interruptPinA), Interrupt_A, CHANGE);
  attachInterrupt(digitalPinToInterrupt(interruptPinB), Interrupt_B, CHANGE);

  Serial.begin(115200);
  while(!Serial);

  delay(1000);
  ti = millis();
  t0 = ti;
}

//
void loop() {
  t1 = millis();

  // Periodo de muestreo
  if(t1 - t0 >= dt) {
    t0 = t0 + dt;

    t = (t1-ti)/1000.0; // t = Tiempo actual




    Serial.print(countA);
    Serial.print('\t');
  
    Serial.print(countB);
    Serial.print('\n');
  }
}
