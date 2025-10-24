import paho.mqtt.client as mqtt
import json
import random
import time

# MQTT broker details
BROKER = "broker.hivemq.com"  # public broker for testing
PORT = 1883
TOPIC = "agri/sensor_data"
QOS = 1  # Quality of Service: 0,1,2

# Function to generate dummy sensor data
def generate_dummy_data():
    data = {
        "temperature_C": round(random.uniform(10, 45), 2),
        "humidity_percent": round(random.uniform(20, 100), 2),
        "pressure_hPa": round(random.uniform(970, 1035), 2),
        "soil_moisture_percent": round(random.uniform(0, 100), 2)
    }
    return data

# Create MQTT client
client = mqtt.Client()

try:
    # Connect to broker
    client.connect(BROKER, PORT, keepalive=60)
    print("Connected to MQTT broker")

    # Infinite loop to send data continuously
    while True:
        sensor_data = generate_dummy_data()
        payload = json.dumps(sensor_data)
        client.publish(TOPIC, payload, qos=QOS)
        print(f"Published: {payload}")
        time.sleep(2)  # interval between messages

except KeyboardInterrupt:
    print("Publisher stopped by user")

except Exception as e:
    print("Error:", e)

finally:
    client.disconnect()
    print("Disconnected from MQTT broker")