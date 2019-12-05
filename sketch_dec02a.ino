int incomingByte = 0;
#include <Servo.h>

Servo myservo;  // create servo object to control a servo
// twelve servo objects can be created on most boards

int pos = 0;    // variable to store the servo position
int count = 0;
 
void setup(){
// Open serial connection.
  Serial.begin(9600);
  myservo.attach(3);  // attaches the servo on pin 3 to the servo object 
}
 
void loop(){
if (Serial.available() > 0) {
 // read the incoming byte:
 incomingByte = Serial.read();
 
 // say what you got:
 Serial.print("I got: "); // ASCII printable characters
 Serial.println(incomingByte, DEC);
 for (pos = 90; pos <= 180; pos += 1) { // goes from 0 degrees to 180 degrees
    // in steps of 1 degree
    myservo.write(pos);              // tell servo to go to position in variable 'pos'
    delay(10);                       // waits 15ms for the servo to reach the position
 }
 delay(5000);
 for (pos = 180; pos >= 90; pos -= 1) { // goes from 180 degrees to 0 degrees
    myservo.write(pos);              // tell servo to go to position in variable 'pos'
    delay(10);                       // waits 15ms for the servo to reach the position
 }
// /*delay(3000);
// for (pos = 0; pos <= 180; pos += 1) { // goes from 0 degrees to 180 degrees
//    // in steps of 1 degree
//    myservo.write(pos);              // tell servo to go to position in variable 'pos'
//    delay(10);                       // waits 15ms for the servo to reach the position
// }*/



} 
 
}
