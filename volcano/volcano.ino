#include <Adafruit_NeoPixel.h>
#include <ESPmDNS.h>
#include <WiFi.h>
#include "AsyncUDP.h"

#define NEOPIXEL_PIN 5
#define BLUE_TOGGLE 21
#define RED_TOGGLE 19
#define VOLCANO_TOGGLE 3
#define VOLCANO_PIN 18

#define VOLCANO_DELAY 3500
#define CONNECT_DELAY 5000
#define NORMAL_DELAY 100
#define NUM_PIXELS 116

const char* ssid = "roboclash24";
const char* password = "captainblackblack";
const String hostname = "RC24-BV";
const unsigned long interval = 1000;

Adafruit_NeoPixel pixels(NUM_PIXELS, NEOPIXEL_PIN, NEO_GRB + NEO_KHZ800);
AsyncUDP udp;
unsigned long previousMillis = 0;
bool prev_red_state = false, prev_blue_state = false;

void blue();
void red();
void erupt();

void setup() {
  pinMode(BLUE_TOGGLE, INPUT_PULLUP);
  pinMode(RED_TOGGLE, INPUT_PULLUP);
  pinMode(VOLCANO_TOGGLE, INPUT_PULLUP);
  pinMode(VOLCANO_PIN, OUTPUT);
  digitalWrite(VOLCANO_PIN, LOW);

  pixels.begin();
  //pixels.setBrightness(20);
  Serial.begin(9600);

  WiFi.mode(WIFI_STA);
  WiFi.config(INADDR_NONE, INADDR_NONE, INADDR_NONE);
  if (!WiFi.setHostname(hostname.c_str())) {
    Serial.println("Hostname failed to configure");
  }
  WiFi.begin(ssid, password);

  while (WiFi.waitForConnectResult() != WL_CONNECTED) {
    Serial.println("Connection Failed! Rebooting...");
    delay(CONNECT_DELAY);
    ESP.restart();
  }
  Serial.println(WiFi.localIP());

  if (udp.listen(9999)) {
    udp.onPacket([](AsyncUDPPacket packet) {
      String msg = packet.readStringUntil('\n');
      Serial.println("Message received: " + msg);

      if (msg == "red") red();
      else if (msg == "blue") blue();
      else if (msg == "off") none();
      else if (msg == "erupt") erupt();
    });
  }
}

void loop() {
  unsigned long currentMillis = millis();

  if ((WiFi.status() != WL_CONNECTED) && (currentMillis - previousMillis >= interval)) {
    Serial.println("Reconnecting to WiFi...");
    WiFi.disconnect();
    WiFi.reconnect();
    previousMillis = currentMillis;
  }
 
  int red_state = digitalRead(RED_TOGGLE);
  int blue_state = digitalRead(BLUE_TOGGLE);
  int volcano_state = digitalRead(VOLCANO_TOGGLE);
  //Serial.println(String(red_state) + " " + String(blue_state) + " " + String(volcano_state) + " " + String(light));

  if (red_state == LOW) {
    if (prev_red_state) none();
    else red();
    prev_red_state = !prev_red_state;
  }
  else if (blue_state == LOW) {
    if (prev_blue_state) none();
    else blue();
    prev_blue_state = !prev_blue_state;
  }
  else if (volcano_state == LOW) erupt();

  delay(NORMAL_DELAY);
}

void blue() {
  for (int pixel=0; pixel<NUM_PIXELS; pixel++) pixels.setPixelColor(pixel, pixels.Color(0, 0, 255));
  Serial.println("Blue light activated");
  pixels.show();
}

void red() {
  for (int pixel=0; pixel<NUM_PIXELS; pixel++) pixels.setPixelColor(pixel, pixels.Color(255, 0, 0));
  Serial.println("Red light activated");
  pixels.show();
}

void none() {
  for (int pixel=0; pixel<NUM_PIXELS; pixel++) pixels.setPixelColor(pixel, pixels.Color(255, 255, 255));
  Serial.println("Light is turned off");
  pixels.show();
}

void erupt() {
  Serial.println("Erupting!");
  digitalWrite(VOLCANO_PIN, HIGH);
  delay(VOLCANO_DELAY);
  digitalWrite(VOLCANO_PIN, LOW);
}
