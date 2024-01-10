#include <WiFi.h>
#include "AsyncUDP.h"
#include <ESPmDNS.h>

const char * ssid = "roboclash24";
const char * password = "captainblackblack";

const int LED_RED = 14;
const int LED_BLUE = 25;
const int COIL_GREEN = 23;
const int LED_GND = 27;

bool disabled = true;

const String id = "R2";
const String color = "R";
const String hostname = "RC24-Relay-" + id;

unsigned long previousMillis = 0;
unsigned long interval = 1000;

AsyncUDP udp;

void relayOFF(void);
void relayON(void);
void init(void);
void startMatch(void);
void stopMatch(void);
void freeze(void);
void indicator(String id);
void setup() {
  pinMode(LED_RED, OUTPUT);
  pinMode(LED_BLUE, OUTPUT); 
  pinMode(COIL_GREEN, OUTPUT); 
  pinMode(LED_GND, OUTPUT);
  digitalWrite(LED_GND, 0);
  Serial.begin(115200);
  indicator(id);
  init();
  WiFi.mode(WIFI_STA);
  WiFi.config(INADDR_NONE, INADDR_NONE, INADDR_NONE);
  WiFi.setHostname(hostname.c_str());  
  WiFi.begin(ssid, password);
  while (WiFi.waitForConnectResult() != WL_CONNECTED) {
    Serial.println("Connection Failed! Rebooting...");
    delay(5000);
    ESP.restart();
  }
  if(udp.listen(9999)) {
    udp.onPacket([](AsyncUDPPacket packet){
      String msg = packet.readStringUntil('\n');
      Serial.println(msg);
      if(msg == "start") startMatch();
      if(msg == "freeze"+id) freeze();
      if(msg == "unfreeze"+id) unfreeze();
      if(msg == "stop") stopMatch();
    });
  }
   

}

void loop() {
  unsigned long currentMillis = millis();
  // if WiFi is down, try reconnecting every CHECK_WIFI_TIME seconds
  if ((WiFi.status() != WL_CONNECTED) && (currentMillis - previousMillis >=interval)) {
    Serial.println("Reconnecting to WiFi...");
    WiFi.disconnect();
    WiFi.reconnect();
    previousMillis = currentMillis;
  }
}

void relayOFF() {
  digitalWrite(COIL_GREEN, 1);
}

void relayON() {
  digitalWrite(COIL_GREEN, 0);
}

void init() {
  disabled = true;
  digitalWrite(LED_RED, 1);
  digitalWrite(LED_BLUE, 0);
  digitalWrite(LED_GND, 0);
  relayOFF();
}

void startMatch() {
  if(disabled){
    disabled = false;
    digitalWrite(LED_RED, 0);
    relayON();
  }
}

void stopMatch() {
  init();
}

void freeze() {
  if(!disabled){
    digitalWrite(LED_BLUE, 1);
    relayOFF();
  }
}

void unfreeze() {
  if(!disabled){
    digitalWrite(LED_BLUE, 0);
    relayON();
  }
}

void indicator(String id){
  if (id == "R1") {
      digitalWrite(LED_RED, 1);
      digitalWrite(LED_BLUE, 0);
      delay(100);
      digitalWrite(LED_RED, 0);
      digitalWrite(LED_BLUE, 0);
      delay(1000);
  } 
  if (id == "R2") {
      digitalWrite(LED_RED, 1);
      digitalWrite(LED_BLUE, 0);
      delay(100);
      digitalWrite(LED_RED, 0);
      digitalWrite(LED_BLUE, 0);
      delay(100);
      digitalWrite(LED_RED, 1);
      digitalWrite(LED_BLUE, 0);
      delay(100);
      digitalWrite(LED_RED, 0);
      digitalWrite(LED_BLUE, 0);
      delay(1000);
  } 
  if (id == "B1") {
      digitalWrite(LED_RED, 0);
      digitalWrite(LED_BLUE, 1);
      delay(100);
      digitalWrite(LED_RED, 0);
      digitalWrite(LED_BLUE, 0);
      delay(1000);
  } 
  if (id == "B2") {
      digitalWrite(LED_RED, 0);
      digitalWrite(LED_BLUE, 1);
      delay(100);
      digitalWrite(LED_RED, 0);
      digitalWrite(LED_BLUE, 0);
      delay(100);
      digitalWrite(LED_RED, 0);
      digitalWrite(LED_BLUE, 1);
      delay(100);
      digitalWrite(LED_RED, 0);
      digitalWrite(LED_BLUE, 0);
      delay(1000);
  }
}
