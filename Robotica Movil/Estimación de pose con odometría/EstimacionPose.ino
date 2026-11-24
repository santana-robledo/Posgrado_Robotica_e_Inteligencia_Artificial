float t;
unsigned long ti, dt = 50, t0, t1; // dt = 50mS, dt = t1 - t0

//
int pinSTBY = 12;
int pwm_l = 0, pwm_r = 0;

float N = 40.0, R = 0.035, L = 0.13;
float x = 0, y = 0, theta = 0;
int cl_old = 0, cr_old = 0;

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

    int cr = countA, cl = countB;
  
    int delta_cl = cl-cl_old;
    cl_old = cl;
    
    int delta_cr = cr-cr_old;
    cr_old = cr;

    float dl = 2*PI*R*(delta_cl/N);
    float dr = 2*PI*R*(delta_cr/N);
  
    float delta_d = (dr+dl)/2;
    float delta_theta = (dr-dl)/L;
    
    x = x + delta_d*cos(theta);
    y = y + delta_d*sin(theta);
    theta = theta + delta_theta;
    theta = atan2(sin(theta),cos(theta));


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


    Serial.print(x);
    Serial.print('\t');
  
    Serial.print(y);
    Serial.print('\t');
  
    Serial.print(theta*(180/PI));
    Serial.print('\n');
  }
}
