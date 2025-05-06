#include <Servo.h>

#define TRIG_PIN  23  // ESP32 pin GPIO23 connected to Ultrasonic Sensor's TRIG pin
#define ECHO_PIN  22  // ESP32 pin GPIO22 connected to Ultrasonic Sensor's ECHO pin
#define SERVO_PIN 26  // ESP32 pin GPIO26 connected to Servo Motor's control pin

Servo servo; // create servo object to control a servo

float duration_us, distance_cm;
float theta_deg, theta_rad;
float x, y;

void setup() {
  Serial.begin(9600);        // initialize serial port
  pinMode(TRIG_PIN, OUTPUT); // set TRIG pin as output
  pinMode(ECHO_PIN, INPUT);  // set ECHO pin as input
  servo.attach(SERVO_PIN);   // attach servo to defined pin
  servo.write(0);            // initialize servo position
}

void loop() {
  // Sweep from 0 to 180 degrees
  for (int angle = 0; angle <= 180; angle += 5) {
    measureAndPrint(angle);
  }

  // Sweep back from 180 to 0 degrees
  for (int angle = 180; angle >= 0; angle -= 5) {
    measureAndPrint(angle);
  }
}

// Function to handle moving the servo, measuring distance, and printing
void measureAndPrint(int angle) {
  servo.write(angle);        // move servo to specified angle
  delay(200);                // allow time for servo to stabilize

  // Reset variables
  duration_us = 0;
  distance_cm = 0;
  x = 0;
  y = 0;

  // Trigger the ultrasonic sensor
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  // Read echo duration
  duration_us = pulseIn(ECHO_PIN, HIGH, 30000); // 30 ms timeout

  if (duration_us == 0) {
    Serial.print("Angle (deg): "); Serial.print(angle);
    Serial.println(", No object detected (timeout)");
    return; // skip further calculations
  }

  // Calculate distance in cm
  distance_cm = 0.017 * duration_us;

  // Calculate x and y
  theta_deg = angle;
  theta_rad = radians(theta_deg);
  x = distance_cm * cos(theta_rad);
  y = distance_cm * sin(theta_rad);

  // Print data
  Serial.print("Angle (deg): "); Serial.print(theta_deg);
  Serial.print(", Distance (cm): "); Serial.print(distance_cm);
  Serial.print(", X (cm): "); Serial.print(x);
  Serial.print(", Y (cm): "); Serial.println(y);
}

