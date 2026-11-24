
float t;
unsigned long ti, dt = 50, t0, t1; // dt = 50mS, dt = t1 - t0

void setup() {

  
  
  Serial.begin(115200);
  while(!Serial);

  delay(1000);
  ti = millis();
  t0 = ti;
}

void loop() {
  t1 = millis();

  // Periodo de muestreo
  if(t1 - t0 >= dt) {
    t0 = t0 + dt;

    t = (t1-ti)/1000.0; // t = Tiempo actual




  
    Serial.println(t);
  }
}
