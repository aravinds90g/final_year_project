#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>

// ── WiFi & Network Config ────────────────────────────────────────────────
const char* WIFI_SSID     = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";

// The IP of the "Bridge Laptop" running sdn/gateway.py
const char* SERVER_URL    = "http://192.168.1.10:3002/api/traffic"; 
const char* DEVICE_ID     = "ESP32-NORMAL-01";

// ── Pin Definitions (ESP32) ──────────────────────────────────────────────
#define LED_BUILTIN 2    // Onboard LED (Status)

void setup() {
  Serial.begin(115200);
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);

  connectWiFi();
}

void connectWiFi() {
  Serial.print("\nConnecting to WiFi: ");
  Serial.println(WIFI_SSID);
  
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
    digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN)); // Blink while connecting
  }
  
  digitalWrite(LED_BUILTIN, HIGH); // Solid ON when connected
  Serial.println("\nWiFi Connected!");
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());
}

String generatePayload() {
  StaticJsonDocument<256> doc;
  
  // Send simple IoT data
  doc["device_id"]   = DEVICE_ID;
  doc["temperature"] = random(220, 260) / 10.0; // e.g. 24.5
  doc["humidity"]    = random(500, 650) / 10.0; // e.g. 60.2
  
  String payload;
  serializeJson(doc, payload);
  return payload;
}

void sendTraffic(String payload) {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi disconnected. Reconnecting...");
    connectWiFi();
    return;
  }

  HTTPClient http;
  http.begin(SERVER_URL);
  http.addHeader("Content-Type", "application/json");

  // Turn off LED to indicate sending
  digitalWrite(LED_BUILTIN, LOW);
  int httpResponseCode = http.POST(payload);
  digitalWrite(LED_BUILTIN, HIGH); // Turn back on

  if (httpResponseCode > 0) {
    Serial.printf("Sent data: %s | HTTP Code: %d\n", payload.c_str(), httpResponseCode);
  } else {
    Serial.printf("Error code: %d. Gateway down or SDN Blocked?\n", httpResponseCode);
  }
  
  http.end();
}

void loop() {
  sendTraffic(generatePayload());

  // NORMAL DEVICE DELAY: Send data once every 5 seconds
  delay(5000); 
}
