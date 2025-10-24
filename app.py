import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.uix.gridlayout import GridLayout
from kivy.uix.scrollview import ScrollView
from kivy.core.window import Window
from kivy.graphics import Color, RoundedRectangle, Line, Ellipse
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import threading
import paho.mqtt.client as mqtt
import pandas as pd
import json
import time
from collections import deque
import joblib
import os

# ---------------- Settings ----------------
BROKER = "broker.hivemq.com"
PORT = 1883
TOPIC = "agri/sensor_data"

# Load Random Forest model
MODEL_PATH = "agri_water_model.pkl"
try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"✅ Model loaded from {MODEL_PATH}")
    else:
        print(f"⚠️ Model file not found: {MODEL_PATH}")
        model = None
except Exception as e:
    print(f"⚠️ Error loading model: {e}")
    model = None

# Set window background color
Window.clearcolor = (0.08, 0.08, 0.12, 1)  # Dark blue-grey theme

# Global data storage
latest_data = {
    "data": None,
    "timestamp": None,
    "connected": False,
    "prediction": "N/A",
    "probability": 0.0,
    "history": {
        "temperature_C": deque(maxlen=50),
        "humidity_percent": deque(maxlen=50),
        "pressure_hPa": deque(maxlen=50),
        "soil_moisture_percent": deque(maxlen=50),
        "timestamps": deque(maxlen=50)
    }
}

# ---------------- MQTT Callbacks ----------------
def on_connect(client, userdata, flags, rc):
    global latest_data
    if rc == 0:
        print("✅ Connected to MQTT Broker!")
        latest_data["connected"] = True
        client.subscribe(TOPIC)
    else:
        print("❌ Connection failed, Code:", rc)
        latest_data["connected"] = False

def on_message(client, userdata, msg):
    global latest_data
    try:
        data = json.loads(msg.payload.decode())
        
        # Store data
        latest_data["data"] = data
        latest_data["timestamp"] = time.time()
        
        # Add to history for charts
        latest_data["history"]["temperature_C"].append(data.get('temperature_C', 0))
        latest_data["history"]["humidity_percent"].append(data.get('humidity_percent', 0))
        latest_data["history"]["pressure_hPa"].append(data.get('pressure_hPa', 0))
        latest_data["history"]["soil_moisture_percent"].append(data.get('soil_moisture_percent', 0))
        latest_data["history"]["timestamps"].append(time.strftime("%H:%M:%S"))
        
        # ML Prediction
        if model is not None:
            try:
                df = pd.DataFrame([data])
                prediction = model.predict(df)[0]
                probability = model.predict_proba(df)[0]
                
                latest_data["prediction"] = "Anomaly" if prediction == 1 else "Normal"
                latest_data["probability"] = round(probability[1] * 100, 1)  # Anomaly probability
            except Exception as e:
                print(f"⚠️ Prediction error: {e}")
                latest_data["prediction"] = "Error"
                latest_data["probability"] = 0.0
        
        print(f"✓ Received: Temp={data.get('temperature_C')}°C | Prediction: {latest_data['prediction']}")
    except Exception as e:
        print("Error in MQTT:", e)

def on_disconnect(client, userdata, rc):
    global latest_data
    print("✗ Disconnected from MQTT broker")
    latest_data["connected"] = False

def start_mqtt():
    """MQTT connection in background thread"""
    while True:
        try:
            client = mqtt.Client(protocol=mqtt.MQTTv311)
            client.on_connect = on_connect
            client.on_message = on_message
            client.on_disconnect = on_disconnect
            client.connect(BROKER, PORT, keepalive=60)
            client.loop_forever()
        except Exception as e:
            print("Reconnecting MQTT in 5s...", e)
            time.sleep(5)

threading.Thread(target=start_mqtt, daemon=True).start()

# ---------------- UI Components ----------------
class GradientCard(BoxLayout):
    """Modern card with gradient-like effect"""
    def __init__(self, color1=(0.15, 0.2, 0.3, 1), **kwargs):
        super().__init__(**kwargs)
        with self.canvas.before:
            Color(*color1)
            self.rect = RoundedRectangle(radius=[20])
        self.bind(pos=self.update_rect, size=self.update_rect)
    
    def update_rect(self, *args):
        self.rect.pos = self.pos
        self.rect.size = self.size
    
    def set_color(self, color):
        self.canvas.before.clear()
        with self.canvas.before:
            Color(*color)
            self.rect = RoundedRectangle(pos=self.pos, size=self.size, radius=[20])

class SensorCard(GradientCard):
    """Premium sensor display card"""
    def __init__(self, title, icon, unit, **kwargs):
        super().__init__(orientation="vertical", padding=20, spacing=8, **kwargs)
        self.unit = unit
        
        # Icon & Title
        header = BoxLayout(orientation='horizontal', size_hint_y=0.3, spacing=10)
        icon_label = Label(text=icon, font_size=28, size_hint_x=0.3)
        title_label = Label(
            text=title,
            font_size=16,
            color=(0.7, 0.8, 0.9, 1),
            halign='left',
            valign='middle'
        )
        title_label.bind(size=title_label.setter('text_size'))
        header.add_widget(icon_label)
        header.add_widget(title_label)
        
        # Large value display
        self.value_label = Label(
            text="-",
            font_size=48,
            color=(1, 1, 1, 1),
            bold=True,
            size_hint_y=0.5
        )
        
        # Status indicator
        self.status_label = Label(
            text="Waiting...",
            font_size=14,
            color=(0.6, 0.6, 0.6, 1),
            size_hint_y=0.2
        )
        
        self.add_widget(header)
        self.add_widget(self.value_label)
        self.add_widget(self.status_label)
    
    def update_value(self, value):
        self.value_label.text = f"{value:.1f}{self.unit}"

class MLPredictionCard(GradientCard):
    """AI Prediction display"""
    def __init__(self, **kwargs):
        super().__init__(orientation="vertical", padding=20, spacing=15, 
                        color1=(0.2, 0.15, 0.25, 1), **kwargs)
        
        # Title
        title = Label(
            text="🤖 AI Anomaly Detection",
            font_size=20,
            color=(1, 0.8, 0.4, 1),
            bold=True,
            size_hint_y=0.2
        )
        
        # Prediction result
        self.prediction_label = Label(
            text="Analyzing...",
            font_size=36,
            color=(0.8, 0.8, 0.8, 1),
            bold=True,
            size_hint_y=0.4
        )
        
        # Probability
        self.prob_label = Label(
            text="Confidence: --%",
            font_size=18,
            color=(0.7, 0.7, 0.7, 1),
            size_hint_y=0.2
        )
        
        # Model info
        model_info = Label(
            text=f"Model: {'Random Forest ✓' if model else 'Not Loaded ✗'}",
            font_size=12,
            color=(0.5, 0.5, 0.5, 1),
            size_hint_y=0.2
        )
        
        self.add_widget(title)
        self.add_widget(self.prediction_label)
        self.add_widget(self.prob_label)
        self.add_widget(model_info)
    
    def update_prediction(self, prediction, probability):
        if prediction == "Anomaly":
            self.prediction_label.text = "⚠️ ANOMALY DETECTED"
            self.prediction_label.color = (1, 0.3, 0.3, 1)
            self.set_color((0.3, 0.15, 0.15, 1))
        elif prediction == "Normal":
            self.prediction_label.text = "✓ NORMAL"
            self.prediction_label.color = (0.3, 1, 0.5, 1)
            self.set_color((0.15, 0.25, 0.15, 1))
        else:
            self.prediction_label.text = prediction
            self.prediction_label.color = (0.8, 0.8, 0.8, 1)
        
        self.prob_label.text = f"Anomaly Probability: {probability}%"

class StatusCard(GradientCard):
    """Connection status"""
    def __init__(self, **kwargs):
        super().__init__(orientation="vertical", padding=20, spacing=10, 
                        color1=(0.15, 0.15, 0.2, 1), **kwargs)
        
        title = Label(
            text="📡 System Status",
            font_size=18,
            color=(1, 1, 1, 1),
            bold=True,
            size_hint_y=0.3
        )
        
        self.status_label = Label(
            text="Disconnected",
            font_size=24,
            color=(1, 0.3, 0.3, 1),
            bold=True,
            size_hint_y=0.4
        )
        
        self.info_label = Label(
            text=f"Broker: {BROKER}\nTopic: {TOPIC}",
            font_size=11,
            color=(0.5, 0.5, 0.5, 1),
            size_hint_y=0.3
        )
        
        self.add_widget(title)
        self.add_widget(self.status_label)
        self.add_widget(self.info_label)
    
    def update_status(self, connected):
        if connected:
            self.status_label.text = "🟢 Connected"
            self.status_label.color = (0.3, 1, 0.5, 1)
            self.set_color((0.15, 0.25, 0.15, 1))
        else:
            self.status_label.text = "🔴 Disconnected"
            self.status_label.color = (1, 0.3, 0.3, 1)
            self.set_color((0.25, 0.15, 0.15, 1))

class LiveChartCard(GradientCard):
    """Live updating chart"""
    def __init__(self, **kwargs):
        super().__init__(orientation="vertical", padding=15, spacing=10, 
                        color1=(0.12, 0.14, 0.18, 1), **kwargs)
        
        title = Label(
            text="📈 Real-time Sensor Trends (Last 50 readings)",
            font_size=18,
            color=(1, 1, 1, 1),
            bold=True,
            size_hint_y=0.08
        )
        self.add_widget(title)
        
        # Create matplotlib figure
        plt.style.use('dark_background')
        self.figure = Figure(figsize=(10, 4), facecolor='#1a1d24')
        self.ax = self.figure.add_subplot(111, facecolor='#1a1d24')
        
        self.canvas_widget = FigureCanvasKivyAgg(self.figure)
        self.add_widget(self.canvas_widget)
        
        # Initial plot setup
        self.setup_plot()
    
    def setup_plot(self):
        self.ax.clear()
        self.ax.set_xlabel('Time', color='white', fontsize=10)
        self.ax.set_ylabel('Values', color='white', fontsize=10)
        self.ax.tick_params(colors='white', labelsize=8)
        self.ax.grid(True, alpha=0.2, color='gray')
        self.ax.legend(loc='upper left', fontsize=8)
    
    def update_chart(self, history):
        try:
            self.ax.clear()
            
            if len(history["temperature_C"]) > 0:
                x_range = range(len(history["temperature_C"]))
                
                # Plot all sensors
                self.ax.plot(x_range, list(history["temperature_C"]), 
                           label='Temperature', color='#ff6b6b', linewidth=2, marker='o', markersize=3)
                self.ax.plot(x_range, list(history["humidity_percent"]), 
                           label='Humidity', color='#4ecdc4', linewidth=2, marker='s', markersize=3)
                self.ax.plot(x_range, list(history["soil_moisture_percent"]), 
                           label='Soil Moisture', color='#95e77d', linewidth=2, marker='^', markersize=3)
                
                # Styling
                self.ax.set_xlabel('Readings', color='white', fontsize=10)
                self.ax.set_ylabel('Values', color='white', fontsize=10)
                self.ax.tick_params(colors='white', labelsize=8)
                self.ax.grid(True, alpha=0.2, color='gray')
                self.ax.legend(loc='upper left', fontsize=9, facecolor='#2a2d34')
                
                # Set background
                self.ax.set_facecolor('#1a1d24')
                self.figure.patch.set_facecolor('#1a1d24')
            
            self.canvas_widget.draw()
        except Exception as e:
            print(f"Chart update error: {e}")

class AgriDashboard(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation='vertical', padding=15, spacing=15, **kwargs)
        
        # Title bar
        title_box = BoxLayout(size_hint_y=0.08, padding=[10, 5])
        title = Label(
            text="🌾 AI-Powered Smart Agriculture Dashboard",
            font_size=32,
            color=(1, 0.85, 0.3, 1),
            bold=True
        )
        title_box.add_widget(title)
        self.add_widget(title_box)
        
        # Top row: 4 sensor cards
        sensor_row = GridLayout(cols=4, spacing=15, size_hint_y=0.22)
        
        self.temp_card = SensorCard("Temperature", "🌡️", "°C")
        self.humid_card = SensorCard("Humidity", "💧", "%")
        self.pressure_card = SensorCard("Pressure", "🌪️", " hPa")
        self.soil_card = SensorCard("Soil Moisture", "🌱", "%")
        
        sensor_row.add_widget(self.temp_card)
        sensor_row.add_widget(self.humid_card)
        sensor_row.add_widget(self.pressure_card)
        sensor_row.add_widget(self.soil_card)
        
        self.add_widget(sensor_row)
        
        # Middle row: ML Prediction + Status
        middle_row = GridLayout(cols=2, spacing=15, size_hint_y=0.18)
        
        self.ml_card = MLPredictionCard()
        self.status_card = StatusCard()
        
        middle_row.add_widget(self.ml_card)
        middle_row.add_widget(self.status_card)
        
        self.add_widget(middle_row)
        
        # Bottom row: Live chart
        self.chart_card = LiveChartCard(size_hint_y=0.52)
        self.add_widget(self.chart_card)
        
        # Update every 1 second
        Clock.schedule_interval(self.update_dashboard, 1)
    
    def update_dashboard(self, dt):
        """Update all dashboard components"""
        data = latest_data.get("data")
        
        if data:
            # Update sensor cards
            self.temp_card.update_value(data.get('temperature_C', 0))
            self.humid_card.update_value(data.get('humidity_percent', 0))
            self.pressure_card.update_value(data.get('pressure_hPa', 0))
            self.soil_card.update_value(data.get('soil_moisture_percent', 0))
            
            # Update ML prediction
            self.ml_card.update_prediction(
                latest_data.get("prediction", "N/A"),
                latest_data.get("probability", 0.0)
            )
            
            # Update connection status
            time_since_last = time.time() - latest_data.get("timestamp", 0)
            connected = latest_data.get("connected", False) and time_since_last < 5
            self.status_card.update_status(connected)
            
            # Update chart
            self.chart_card.update_chart(latest_data["history"])
        else:
            self.status_card.update_status(False)

class AgriApp(App):
    def build(self):
        Window.size = (1400, 900)
        return AgriDashboard()

if __name__ == "__main__":
    # Install matplotlib backend for Kivy if needed
    try:
        from kivy.garden.matplotlib import backend_kivyagg
    except ImportError:
        print("Installing kivy-garden.matplotlib...")
        import os
        os.system("pip install kivy-garden matplotlib")
        os.system("garden install matplotlib")

    AgriApp().run()