float t;
unsigned long ti, dt = 50, t0, t1; // dt = 50mS, dt = t1 - t0

//
int pinSTBY = 12;
int pwm_l = 0, pwm_r = 0;

// Motor R
int pinPWMA = 6;
int pinAIN2 = 7;
int pinAIN1 = 8;
int interruptPinA = 2;
int countA = 0;

// Motor L
int pinBIN2 = 9;
int pinBIN1 = 10;
int pinPWMB = 11;
int interruptPinB = 3;
int countB = 0;

//
void Interrupt_A () {
  if(pwm_r>=0)
    countA++;
  else
    countA--;
}

void Interrupt_B () {
  if(pwm_l>=0)
    countB++;
  else
    countB--;
}

//
void setup() {
  pinMode(pinAIN2,OUTPUT);
  pinMode(pinAIN1,OUTPUT);
  pinMode(pinPWMA,OUTPUT);
  pinMode(pinBIN1,OUTPUT);
  pinMode(pinBIN2,OUTPUT);
  pinMode(pinPWMB,OUTPUT);
  pinMode(pinSTBY,OUTPUT);
  digitalWrite(pinSTBY, HIGH);
  
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




    pwm_l = 0;
    pwm_r = 0;
  
    if(pwm_l>=0) {
      if(pwm_l>255)
        pwm_l = 255;
        
      digitalWrite(pinAIN2,HIGH);
      digitalWrite(pinAIN1,LOW);
      analogWrite(pinPWMA,pwm_l);
    } else {
      if(pwm_l<-255)
        pwm_l = -255;
  
      digitalWrite(pinAIN2,LOW);
      digitalWrite(pinAIN1,HIGH);
      analogWrite(pinPWMA,-pwm_l);
    }
  
    if(pwm_r>=0) {
       if(pwm_r>255)
        pwm_r = 255;
  
      digitalWrite(pinBIN2,HIGH);
      digitalWrite(pinBIN1,LOW);
      analogWrite(pinPWMB,pwm_r);
    } else {
      if(pwm_r<-255)
        pwm_r = -255;
  
      digitalWrite(pinBIN2,LOW);
      digitalWrite(pinBIN1,HIGH);
      analogWrite(pinPWMB,-pwm_r);
    }


    Serial.print(countA);
    Serial.print('\t');
  
    Serial.print(countB);
    Serial.print('\n');
  }
}
