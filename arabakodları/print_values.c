// print_values.c
#include <stdio.h>
#include <wiringPi.h>
#include <stdio.h>
#include <unistd.h>

#define motor1_ena 25
#define motor1_in1 24
#define motor1_in2 23
#define motor2_ena 27
#define motor2_in3 29
#define motor2_in4 28

void setupMotor(int ena, int in1, int in2) {
    pinMode(ena, OUTPUT);
    pinMode(in1, OUTPUT);
    pinMode(in2, OUTPUT);
    digitalWrite(ena, LOW);
    digitalWrite(in1, LOW);
    digitalWrite(in2, LOW);
}

void forward(int ena, int in1, int in2) {
    digitalWrite(ena, HIGH);
    digitalWrite(in1, HIGH);
    digitalWrite(in2, LOW);
}

void backward(int ena, int in1, int in2) {
    digitalWrite(ena, HIGH);
    digitalWrite(in1, LOW);
    digitalWrite(in2, HIGH);
}

void left(int ena, int in1, int in2) {
    digitalWrite(ena, HIGH);
    digitalWrite(in1, LOW);
    digitalWrite(in2, LOW);
}

void right(int ena, int in1, int in2) {
    digitalWrite(ena, HIGH);
    digitalWrite(in1, HIGH);
    digitalWrite(in2, HIGH);
}

void stop(int ena, int in1, int in2) {
    digitalWrite(ena, LOW);
    digitalWrite(in1, LOW);
    digitalWrite(in2, LOW);
}


void print_values(int x, int y, int scale) {
    wiringPiSetup();
    setupMotor(motor1_ena, motor1_in1, motor1_in2);
    setupMotor(motor2_ena, motor2_in3, motor2_in4);
    
    if (scale > 150) {
        printf("backward\n");
        backward(motor1_ena, motor1_in1, motor1_in2);
        backward(motor2_ena, motor2_in3, motor2_in4);
        
    } else if (scale <= 150) {
        printf("forward\n");
        forward(motor1_ena, motor1_in1, motor1_in2);
        forward(motor2_ena, motor2_in3, motor2_in4);
    } else {
        printf("<<<  wrong data  >>>\n");
        printf("please enter the defined data to continue.....\n");
    }
}


/*


int main(void) {
    printf("Raspberry Pi Dual Motor Control with L298N\n");

    if (wiringPiSetup() == -1) {
        return 1;
    }
    

    

    printf("The default direction of both motors is Forward.\n");
    printf("w-forward s-backward a-left d-right z-stop q-exit\n\n");

while (1) {
    int dummy_x = 0;  // veya istediğiniz bir değeri atayın
    print_values(dummy_x, 0, 0);
}


    return 0;
}
*/
