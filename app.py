import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.clock import Clock
from kivy.uix.label import Label
import threading
import paho.mqtt.client as mqtt
import pandas as pd
import json
import joblib
import time

# MQTT settings
BROKER = "broker.hivemq.com"
PORT = 1883
TOPIC = "agri/sensor_data"

# Load your trained model
model = joblib.load("agri_water_model.pkl")  # <-- update path

# Shared data
latest_data = {"data": None, "prediction": "-", "prob": "-", "timestamp": None}

# ---------------- MQTT CALLBACKS ----------------
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("MQTT Connected successfully!")
        client.subscribe(TOPIC)
    else:
        print("MQTT Connection failed with code:", rc)

def on_message(client, userdata, msg):
    global latest_data
    try:
        payload = msg.payload.decode()
        print(f"Message received on {msg.topic}: {payload}")  # debug
        data = json.loads(payload)
        df = pd.DataFrame([data])
        prediction = model.predict(df)[0]
        prob = model.predict_proba(df)[0][1]

        latest_data["data"] = data
        latest_data["prediction"] = "Anomaly 🚨" if prediction == 1 else "Normal ✅"
        latest_data["prob"] = round(prob, 2)
        latest_data["timestamp"] = time.time()  # last update time
    except Exception as e:
        print("Error in on_message:", e)

# ---------------- MQTT THREAD ----------------
def start_mqtt():
    while True:
        try:
            client = mqtt.Client(protocol=mqtt.MQTTv311)
            client.on_connect = on_connect
            client.on_message = on_message
            client.connect(BROKER, PORT, keepalive=60)
            client.loop_forever()
        except Exception as e:
            print("MQTT connection failed, retrying in 5s:", e)
            time.sleep(5)  # retry on failure

# Start MQTT in background
threading.Thread(target=start_mqtt, daemon=True).start()

# ---------------- KIVY APP ----------------
class AgriDashboard(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation='vertical', padding=20, spacing=15, **kwargs)

        self.data_label = Label(text="Waiting for data...", font_size=18)
        self.pred_label = Label(text="Prediction: -", font_size=22)
        self.prob_label = Label(text="Probability: -", font_size=20)
        self.status_label = Label(text="Status: Disconnected ❌", font_size=18, color=(1,0,0,1))

        self.add_widget(self.data_label)
        self.add_widget(self.pred_label)
        self.add_widget(self.prob_label)
        self.add_widget(self.status_label)

        # Update UI every second
        Clock.schedule_interval(self.update_display, 1)

    def update_display(self, dt):
        if latest_data["data"]:
            self.data_label.text = f"Sensor Data:\n{json.dumps(latest_data['data'], indent=2)}"

            # Prediction color
            if "Anomaly" in latest_data["prediction"]:
                self.pred_label.color = (1,0,0,1)  # Red
            else:
                self.pred_label.color = (0,1,0,1)  # Green
            self.pred_label.text = f"Prediction: {latest_data['prediction']}"
            self.prob_label.text = f"Probability: {latest_data['prob']}"

            # Check if last message was within 5 seconds
            if time.time() - latest_data["timestamp"] <= 5:
                self.status_label.text = "Status: Connected ✅"
                self.status_label.color = (0,1,0,1)
            else:
                self.status_label.text = "Status: Disconnected ❌"
                self.status_label.color = (1,0,0,1)
        else:
            self.data_label.text = "Waiting for data..."
            self.pred_label.text = "Prediction: -"
            self.prob_label.text = "Probability: -"
            self.status_label.text = "Status: Disconnected ❌"
            self.status_label.color = (1,0,0,1)

class AgriApp(App):
    def build(self):
        return AgriDashboard()

if __name__ == "__main__":
    AgriApp().run()
