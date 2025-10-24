from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.uix.gridlayout import GridLayout
from kivy.uix.scrollview import ScrollView
from kivy.core.window import Window
from kivy.graphics import Color, RoundedRectangle, Line, Ellipse, Rectangle
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
from kivy.core.text import LabelBase
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

# Register Font Awesome (download fontawesome-webfont.ttf and place in app directory)
# Download from: https://github.com/FortAwesome/Font-Awesome/blob/fa-4/fonts/fontawesome-webfont.ttf
try:
    LabelBase.register(name='FontAwesome', 
                      fn_regular='fontawesome-webfont.ttf')
    FONT_AWESOME_AVAILABLE = True
except:
    print("⚠️ Font Awesome not found. Download fontawesome-webfont.ttf to app directory")
    FONT_AWESOME_AVAILABLE = False

# Font Awesome icon codes
ICONS = {
    'temperature': '\uf2c9',  # thermometer-half
    'humidity': '\uf043',      # tint/droplet
    'pressure': '\uf0c2',      # cloud
    'soil': '\uf06c',          # leaf
    'leaf': '\uf06c',          # add this line
    'robot': '\uf544',         # robot
    'broadcast': '\uf012',     # signal
    'chart': '\uf201',         # line-chart
    'check': '\uf00c',         # check
    'warning': '\uf071',       # exclamation-triangle
    'circle': '\uf111',        # circle
    'water': '\uf773',   # added water

}

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

# Enhanced color palette
COLORS = {
    'bg_dark': (0.05, 0.05, 0.08, 1),
    'card_bg': (0.12, 0.13, 0.17, 1),
    'card_hover': (0.15, 0.16, 0.20, 1),
    'accent_blue': (0.26, 0.51, 0.96, 1),
    'accent_green': (0.18, 0.80, 0.44, 1),
    'accent_orange': (0.98, 0.55, 0.25, 1),
    'accent_purple': (0.61, 0.35, 0.71, 1),
    'text_primary': (0.95, 0.95, 0.97, 1),
    'text_secondary': (0.65, 0.67, 0.72, 1),
    'text_muted': (0.45, 0.47, 0.52, 1),
    'error': (0.96, 0.26, 0.21, 1),
    'success': (0.18, 0.80, 0.44, 1),
    'warning': (0.98, 0.74, 0.02, 1),
}

# Set window background
Window.clearcolor = COLORS['bg_dark']

# Global data storage
latest_data = {
    "data": None,
    "timestamp": None,
    "connected": False,
    "prediction": "N/A",
    "probability": 0.0,
    "water_needed": False,
    "history": {
        "temperature_C": deque(maxlen=50),
        "humidity_percent": deque(maxlen=50),
        "pressure_hPa": deque(maxlen=50),
        "soil_moisture_percent": deque(maxlen=50),
        "timestamps": deque(maxlen=50)
    }
}

# Water requirement threshold
SOIL_MOISTURE_THRESHOLD = 30.0  # Below this percentage, water is needed

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
        
        latest_data["data"] = data
        latest_data["timestamp"] = time.time()
        
        latest_data["history"]["temperature_C"].append(data.get('temperature_C', 0))
        latest_data["history"]["humidity_percent"].append(data.get('humidity_percent', 0))
        latest_data["history"]["pressure_hPa"].append(data.get('pressure_hPa', 0))
        latest_data["history"]["soil_moisture_percent"].append(data.get('soil_moisture_percent', 0))
        latest_data["history"]["timestamps"].append(time.strftime("%H:%M:%S"))
        
        # Check if water is needed
        soil_moisture = data.get('soil_moisture_percent', 100)
        latest_data["water_needed"] = soil_moisture < SOIL_MOISTURE_THRESHOLD
        
        if model is not None:
            try:
                df = pd.DataFrame([data])
                prediction = model.predict(df)[0]
                probability = model.predict_proba(df)[0]
                
                latest_data["prediction"] = "Anomaly" if prediction == 1 else "Normal"
                latest_data["probability"] = round(probability[1] * 100, 1)
            except Exception as e:
                print(f"⚠️ Prediction error: {e}")
                latest_data["prediction"] = "Error"
                latest_data["probability"] = 0.0
        
        print(f"✓ Received: Temp={data.get('temperature_C')}°C | Prediction: {latest_data['prediction']} | Water: {'NEEDED' if latest_data['water_needed'] else 'OK'}")
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
class ModernCard(BoxLayout):
    """Enhanced card with shadow effect"""
    def __init__(self, bg_color=None, **kwargs):
        super().__init__(**kwargs)
        if bg_color is None:
            bg_color = COLORS['card_bg']
        
        with self.canvas.before:
            # Shadow effect
            Color(0, 0, 0, 0.3)
            self.shadow = RoundedRectangle(radius=[15])
            # Main card
            Color(*bg_color)
            self.rect = RoundedRectangle(radius=[15])
        
        self.bind(pos=self.update_graphics, size=self.update_graphics)
        self.bg_color = bg_color
    
    def update_graphics(self, *args):
        self.shadow.pos = (self.pos[0] + 2, self.pos[1] - 2)
        self.shadow.size = self.size
        self.rect.pos = self.pos
        self.rect.size = self.size
    
    def set_color(self, color):
        self.bg_color = color
        self.canvas.before.clear()
        with self.canvas.before:
            Color(0, 0, 0, 0.3)
            self.shadow = RoundedRectangle(pos=(self.pos[0] + 2, self.pos[1] - 2), 
                                          size=self.size, radius=[15])
            Color(*color)
            self.rect = RoundedRectangle(pos=self.pos, size=self.size, radius=[15])

class IconLabel(Label):
    """Label with Font Awesome icon support"""
    def __init__(self, icon_code='', **kwargs):
        super().__init__(**kwargs)
        if FONT_AWESOME_AVAILABLE and icon_code:
            self.font_name = 'FontAwesome'
            self.text = icon_code
        else:
            # Fallback to text icon
            self.text = kwargs.get('text', icon_code)

class SensorCard(ModernCard):
    """Enhanced sensor display card"""
    def __init__(self, title, icon_code, unit, accent_color, **kwargs):
        super().__init__(orientation="vertical", padding=20, spacing=12, **kwargs)
        self.unit = unit
        self.accent_color = accent_color
        
        # Header with icon
        header = BoxLayout(orientation='horizontal', size_hint_y=0.25, spacing=15)
        
        # Icon with accent background
        icon_container = BoxLayout(size_hint_x=0.35)
        with icon_container.canvas.before:
            Color(*accent_color)
            self.icon_bg = Ellipse(size=(50, 50))
        icon_container.bind(pos=self.update_icon_bg, size=self.update_icon_bg)
        
        self.icon_label = IconLabel(
            icon_code=icon_code,
            font_size=26,
            color=COLORS['text_primary'],
            halign='center',
            valign='middle'
        )
        self.icon_label.bind(size=self.icon_label.setter('text_size'))
        icon_container.add_widget(self.icon_label)
        
        # Title
        title_label = Label(
            text=title,
            font_size=15,
            color=COLORS['text_secondary'],
            halign='left',
            valign='middle',
            bold=True
        )
        title_label.bind(size=title_label.setter('text_size'))
        
        header.add_widget(icon_container)
        header.add_widget(title_label)
        
        # Value display
        self.value_label = Label(
            text="--",
            font_size=52,
            color=COLORS['text_primary'],
            bold=True,
            size_hint_y=0.55
        )
        
        # Status bar
        self.status_label = Label(
            text="Initializing...",
            font_size=13,
            color=COLORS['text_muted'],
            size_hint_y=0.2
        )
        
        self.add_widget(header)
        self.add_widget(self.value_label)
        self.add_widget(self.status_label)
        self.icon_container = icon_container
    
    def update_icon_bg(self, instance, value):
        # Center the circle
        center_x = instance.center_x - 25
        center_y = instance.center_y - 25
        self.icon_bg.pos = (center_x, center_y)
    
    def update_value(self, value):
        self.value_label.text = f"{value:.1f}{self.unit}"
        self.status_label.text = "Live Data"
        self.status_label.color = self.accent_color

class MLPredictionCard(ModernCard):
    """AI Prediction display with enhanced styling"""
    def __init__(self, **kwargs):
        super().__init__(orientation="vertical", padding=25, spacing=18, **kwargs)
        
        # Header
        header = BoxLayout(orientation='horizontal', size_hint_y=0.2, spacing=12)
        
        icon = IconLabel(
            icon_code=ICONS['robot'],
            font_size=28,
            color=COLORS['accent_purple'],
            size_hint_x=0.15
        )
        
        title = Label(
            text="AI Anomaly Detection",
            font_size=20,
            color=COLORS['text_primary'],
            bold=True,
            halign='left',
            valign='middle'
        )
        title.bind(size=title.setter('text_size'))
        
        header.add_widget(icon)
        header.add_widget(title)
        
        # Prediction status
        self.prediction_label = Label(
            text="ANALYZING",
            font_size=38,
            color=COLORS['text_secondary'],
            bold=True,
            size_hint_y=0.35
        )
        
        # Probability bar container
        prob_container = BoxLayout(orientation='vertical', size_hint_y=0.25, spacing=8)
        
        self.prob_label = Label(
            text="Confidence: --%",
            font_size=16,
            color=COLORS['text_secondary']
        )
        
        # Progress bar background
        self.progress_bg = BoxLayout(size_hint_y=0.4)
        with self.progress_bg.canvas.before:
            Color(0.2, 0.2, 0.25, 1)
            self.progress_bg_rect = RoundedRectangle(radius=[10])
        self.progress_bg.bind(pos=self.update_progress_bg, size=self.update_progress_bg)
        
        # Progress bar fill
        self.progress_fill = BoxLayout(size_hint=(0, 1))
        with self.progress_fill.canvas.before:
            Color(*COLORS['accent_blue'])
            self.progress_fill_rect = RoundedRectangle(radius=[10])
        self.progress_fill.bind(pos=self.update_progress_fill, size=self.update_progress_fill)
        
        self.progress_bg.add_widget(self.progress_fill)
        
        prob_container.add_widget(self.prob_label)
        prob_container.add_widget(self.progress_bg)
        
        # Model info
        model_status = "LOADED" if model else "NOT LOADED"
        model_icon = ICONS['check'] if model else ICONS['warning']
        model_color = COLORS['success'] if model else COLORS['error']
        
        model_info_box = BoxLayout(orientation='horizontal', size_hint_y=0.2, spacing=8)
        
        model_icon_label = IconLabel(
            icon_code=model_icon,
            font_size=14,
            color=model_color,
            size_hint_x=0.1
        )
        
        model_info = Label(
            text=f"Model: Random Forest | Status: {model_status}",
            font_size=12,
            color=COLORS['text_muted'],
            halign='left',
            valign='middle'
        )
        model_info.bind(size=model_info.setter('text_size'))
        
        model_info_box.add_widget(model_icon_label)
        model_info_box.add_widget(model_info)
        
        self.add_widget(header)
        self.add_widget(self.prediction_label)
        self.add_widget(prob_container)
        self.add_widget(model_info_box)
    
    def update_progress_bg(self, instance, value):
        self.progress_bg_rect.pos = instance.pos
        self.progress_bg_rect.size = instance.size
    
    def update_progress_fill(self, instance, value):
        self.progress_fill_rect.pos = instance.pos
        self.progress_fill_rect.size = instance.size
    
    def update_prediction(self, prediction, probability):
        if prediction == "Anomaly":
            self.prediction_label.text = "ANOMALY DETECTED"
            self.prediction_label.color = COLORS['error']
            self.set_color((0.25, 0.12, 0.12, 1))
            fill_color = COLORS['error']
        elif prediction == "Normal":
            self.prediction_label.text = "NORMAL OPERATION"
            self.prediction_label.color = COLORS['success']
            self.set_color((0.12, 0.22, 0.15, 1))
            fill_color = COLORS['success']
        else:
            self.prediction_label.text = prediction.upper()
            self.prediction_label.color = COLORS['text_secondary']
            fill_color = COLORS['accent_blue']
        
        self.prob_label.text = f"Anomaly Probability: {probability}%"
        
        # Update progress bar
        self.progress_fill.size_hint_x = float(probability) / 100
        self.progress_fill.canvas.before.clear()
        with self.progress_fill.canvas.before:
            Color(*fill_color)
            self.progress_fill_rect = RoundedRectangle(
                pos=self.progress_fill.pos,
                size=self.progress_fill.size,
                radius=[10]
            )

class StatusCard(ModernCard):
    """Enhanced connection status card"""
    def __init__(self, **kwargs):
        super().__init__(orientation="vertical", padding=25, spacing=15, **kwargs)
        
        # Header
        header = BoxLayout(orientation='horizontal', size_hint_y=0.25, spacing=12)
        
        self.status_icon = IconLabel(
            icon_code=ICONS['broadcast'],
            font_size=28,
            color=COLORS['error'],
            size_hint_x=0.15
        )
        
        title = Label(
            text="System Status",
            font_size=20,
            color=COLORS['text_primary'],
            bold=True,
            halign='left',
            valign='middle'
        )
        title.bind(size=title.setter('text_size'))
        
        header.add_widget(self.status_icon)
        header.add_widget(title)
        
        # Connection status
        self.status_label = Label(
            text="DISCONNECTED",
            font_size=32,
            color=COLORS['error'],
            bold=True,
            size_hint_y=0.3
        )
        
        # Connection indicator circle
        indicator_box = BoxLayout(orientation='horizontal', size_hint_y=0.15, spacing=10)
        
        self.indicator = BoxLayout(size_hint=(None, None), size=(16, 16))
        with self.indicator.canvas.before:
            Color(*COLORS['error'])
            self.indicator_circle = Ellipse(size=(16, 16))
        self.indicator.bind(pos=self.update_indicator, size=self.update_indicator)
        
        indicator_label = Label(
            text="Connection Status",
            font_size=13,
            color=COLORS['text_muted'],
            halign='left',
            valign='middle'
        )
        indicator_label.bind(size=indicator_label.setter('text_size'))
        
        indicator_box.add_widget(self.indicator)
        indicator_box.add_widget(indicator_label)
        
        # Info labels
        self.info_label = Label(
            text=f"Broker: {BROKER}\nTopic: {TOPIC}",
            font_size=11,
            color=COLORS['text_muted'],
            size_hint_y=0.15,
            halign='center',
            valign='middle'
        )
        self.info_label.bind(size=self.info_label.setter('text_size'))
        
        self.add_widget(header)
        self.add_widget(self.status_label)
        self.add_widget(indicator_box)
        self.add_widget(self.info_label)
    
    def update_indicator(self, instance, value):
        self.indicator_circle.pos = instance.pos
    
    def update_status(self, connected):
        if connected:
            self.status_label.text = "CONNECTED"
            self.status_label.color = COLORS['success']
            self.status_icon.color = COLORS['success']
            self.set_color((0.12, 0.22, 0.15, 1))
            
            self.indicator.canvas.before.clear()
            with self.indicator.canvas.before:
                Color(*COLORS['success'])
                self.indicator_circle = Ellipse(pos=self.indicator.pos, size=(16, 16))
        else:
            self.status_label.text = "DISCONNECTED"
            self.status_label.color = COLORS['error']
            self.status_icon.color = COLORS['error']
            self.set_color((0.22, 0.12, 0.12, 1))
            
            self.indicator.canvas.before.clear()
            with self.indicator.canvas.before:
                Color(*COLORS['error'])
                self.indicator_circle = Ellipse(pos=self.indicator.pos, size=(16, 16))


class WaterAlertCard(ModernCard):
    """Water requirement alert card"""
    def __init__(self, **kwargs):
        super().__init__(orientation="vertical", padding=25, spacing=15, **kwargs)
        
        # Header with animated water icon
        header = BoxLayout(orientation='horizontal', size_hint_y=0.25, spacing=12)
        
        self.water_icon = IconLabel(
            icon_code=ICONS['water'],
            font_size=32,
            color=COLORS['accent_blue'],
            size_hint_x=0.15
        )
        
        title = Label(
            text="Water Status",
            font_size=20,
            color=COLORS['text_primary'],
            bold=True,
            halign='left',
            valign='middle'
        )
        title.bind(size=title.setter('text_size'))
        
        header.add_widget(self.water_icon)
        header.add_widget(title)
        
        # Main status label
        self.status_label = Label(
            text="CHECKING...",
            font_size=28,
            color=COLORS['text_secondary'],
            bold=True,
            size_hint_y=0.35
        )
        
        # Soil moisture level
        self.moisture_label = Label(
            text="Soil Moisture: --%",
            font_size=16,
            color=COLORS['text_secondary'],
            size_hint_y=0.2
        )
        
        # Recommendation text
        self.recommendation_label = Label(
            text="Waiting for data...",
            font_size=13,
            color=COLORS['text_muted'],
            size_hint_y=0.2,
            halign='center',
            valign='middle'
        )
        self.recommendation_label.bind(size=self.recommendation_label.setter('text_size'))
        
        self.add_widget(header)
        self.add_widget(self.status_label)
        self.add_widget(self.moisture_label)
        self.add_widget(self.recommendation_label)
    
    def update_water_status(self, water_needed, soil_moisture):
        self.moisture_label.text = f"Soil Moisture: {soil_moisture:.1f}%"
        
        if water_needed:
            self.status_label.text = "WATER NEEDED"
            self.status_label.color = COLORS['warning']
            self.water_icon.color = COLORS['warning']
            self.set_color((0.25, 0.20, 0.10, 1))
            self.recommendation_label.text = f"Soil moisture below {SOIL_MOISTURE_THRESHOLD}%. Irrigation recommended."
            self.recommendation_label.color = COLORS['warning']
        else:
            self.status_label.text = "OPTIMAL"
            self.status_label.color = COLORS['success']
            self.water_icon.color = COLORS['accent_blue']
            self.set_color((0.12, 0.22, 0.15, 1))
            self.recommendation_label.text = "Soil moisture is at healthy levels. No watering needed."
            self.recommendation_label.color = COLORS['success']

class LiveChartCard(ModernCard):
    """Enhanced live chart with modern styling"""
    def __init__(self, **kwargs):
        super().__init__(orientation="vertical", padding=20, spacing=12, **kwargs)
        
        # Header
        header = BoxLayout(orientation='horizontal', size_hint_y=0.08, spacing=12)
        
        icon = IconLabel(
            icon_code=ICONS['chart'],
            font_size=24,
            color=COLORS['accent_blue'],
            size_hint_x=0.05
        )
        
        title = Label(
            text="Real-time Sensor Trends (Last 50 readings)",
            font_size=18,
            color=COLORS['text_primary'],
            bold=True,
            halign='left',
            valign='middle'
        )
        title.bind(size=title.setter('text_size'))
        
        header.add_widget(icon)
        header.add_widget(title)
        
        self.add_widget(header)
        
        # Matplotlib chart
        plt.style.use('dark_background')
        self.figure = Figure(figsize=(12, 5), facecolor='#1a1d24')
        self.ax = self.figure.add_subplot(111, facecolor='#0d0e11')
        
        self.canvas_widget = FigureCanvasKivyAgg(self.figure)
        self.add_widget(self.canvas_widget)
        
        self.setup_plot()
    
    def setup_plot(self):
        self.ax.clear()
        self.ax.set_xlabel('Time', color='#a5a8b0', fontsize=11, fontweight='bold')
        self.ax.set_ylabel('Values', color='#a5a8b0', fontsize=11, fontweight='bold')
        self.ax.tick_params(colors='#6b6e76', labelsize=9)
        self.ax.grid(True, alpha=0.15, color='#3a3d45', linestyle='--', linewidth=0.5)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['left'].set_color('#3a3d45')
        self.ax.spines['bottom'].set_color('#3a3d45')
    
    def update_chart(self, history):
        try:
            self.ax.clear()
            
            if len(history["temperature_C"]) > 0:
                x_range = range(len(history["temperature_C"]))
                
                # Enhanced plot styling
                self.ax.plot(x_range, list(history["temperature_C"]), 
                           label='Temperature', color='#4285f4', linewidth=2.5, 
                           marker='o', markersize=4, alpha=0.9)
                self.ax.plot(x_range, list(history["humidity_percent"]), 
                           label='Humidity', color='#34a853', linewidth=2.5, 
                           marker='s', markersize=4, alpha=0.9)
                self.ax.plot(x_range, list(history["soil_moisture_percent"]), 
                           label='Soil Moisture', color='#fbbc04', linewidth=2.5, 
                           marker='^', markersize=4, alpha=0.9)
                
                # Styling
                self.ax.set_xlabel('Readings', color='#a5a8b0', fontsize=11, fontweight='bold')
                self.ax.set_ylabel('Values', color='#a5a8b0', fontsize=11, fontweight='bold')
                self.ax.tick_params(colors='#6b6e76', labelsize=9)
                self.ax.grid(True, alpha=0.15, color='#3a3d45', linestyle='--', linewidth=0.5)
                
                legend = self.ax.legend(loc='upper left', fontsize=10, 
                                       facecolor='#1a1d24', edgecolor='#3a3d45',
                                       framealpha=0.95)
                for text in legend.get_texts():
                    text.set_color('#a5a8b0')
                
                # Remove top and right spines
                self.ax.spines['top'].set_visible(False)
                self.ax.spines['right'].set_visible(False)
                self.ax.spines['left'].set_color('#3a3d45')
                self.ax.spines['bottom'].set_color('#3a3d45')
                
                self.ax.set_facecolor('#0d0e11')
                self.figure.patch.set_facecolor('#1a1d24')
            
            self.canvas_widget.draw()
        except Exception as e:
            print(f"Chart update error: {e}")

class AgriDashboard(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation='vertical', padding=20, spacing=18, **kwargs)
        
        # Header bar with gradient-like effect
        header_container = BoxLayout(size_hint_y=0.09, padding=[15, 10])
        
        with header_container.canvas.before:
            Color(0.08, 0.09, 0.12, 1)
            self.header_bg = RoundedRectangle(radius=[12])
        header_container.bind(pos=self.update_header_bg, size=self.update_header_bg)
        
        header = BoxLayout(orientation='horizontal', spacing=15)
        
        # App icon
        app_icon = IconLabel(
            icon_code=ICONS['leaf'],
            font_size=36,
            color=COLORS['accent_green'],
            size_hint_x=0.08
        )
        
        # Title
        title = Label(
            text="Smart Agriculture Monitoring System",
            font_size=32,
            color=COLORS['text_primary'],
            bold=True,
            halign='left',
            valign='middle'
        )
        title.bind(size=title.setter('text_size'))
        
        # Subtitle
        subtitle = Label(
            text="AI-Powered Real-time Analytics",
            font_size=14,
            color=COLORS['accent_green'],
            halign='right',
            valign='middle',
            size_hint_x=0.3
        )
        subtitle.bind(size=subtitle.setter('text_size'))
        
        header.add_widget(app_icon)
        header.add_widget(title)
        header.add_widget(subtitle)
        header_container.add_widget(header)
        
        self.add_widget(header_container)
        
        # Sensor cards grid
        sensor_row = GridLayout(cols=4, spacing=18, size_hint_y=0.24)
        
        self.temp_card = SensorCard("Temperature", ICONS['temperature'], "°C", 
                                    COLORS['accent_orange'])
        self.humid_card = SensorCard("Humidity", ICONS['humidity'], "%", 
                                     COLORS['accent_blue'])
        self.pressure_card = SensorCard("Pressure", ICONS['pressure'], " hPa", 
                                       COLORS['accent_purple'])
        self.soil_card = SensorCard("Soil Moisture", ICONS['soil'], "%", 
                                    COLORS['accent_green'])
        
        sensor_row.add_widget(self.temp_card)
        sensor_row.add_widget(self.humid_card)
        sensor_row.add_widget(self.pressure_card)
        sensor_row.add_widget(self.soil_card)
        
        self.add_widget(sensor_row)
        
        # Middle row: ML + Status + Water Alert (3 columns now)
        middle_row = GridLayout(cols=3, spacing=18, size_hint_y=0.20)
        
        self.ml_card = MLPredictionCard()
        self.status_card = StatusCard()
        self.water_card = WaterAlertCard()
        
        middle_row.add_widget(self.ml_card)
        middle_row.add_widget(self.status_card)
        middle_row.add_widget(self.water_card)
        
        self.add_widget(middle_row)
        
        # Chart
        self.chart_card = LiveChartCard(size_hint_y=0.47)
        self.add_widget(self.chart_card)
        
        # Update interval
        Clock.schedule_interval(self.update_dashboard, 1)
    
    def update_header_bg(self, instance, value):
        self.header_bg.pos = instance.pos
        self.header_bg.size = instance.size
    
    def update_dashboard(self, dt):
        """Update all dashboard components"""
        data = latest_data.get("data")
        
        if data:
            self.temp_card.update_value(data.get('temperature_C', 0))
            self.humid_card.update_value(data.get('humidity_percent', 0))
            self.pressure_card.update_value(data.get('pressure_hPa', 0))
            self.soil_card.update_value(data.get('soil_moisture_percent', 0))
            
            self.ml_card.update_prediction(
                latest_data.get("prediction", "N/A"),
                latest_data.get("probability", 0.0)
            )
            
            time_since_last = time.time() - latest_data.get("timestamp", 0)
            connected = latest_data.get("connected", False) and time_since_last < 5
            self.status_card.update_status(connected)
            
            self.chart_card.update_chart(latest_data["history"])
        else:
            self.status_card.update_status(False)

class AgriApp(App):
    def build(self):
        Window.size = (1600, 950)
        self.title = "Smart Agriculture Dashboard"
        return AgriDashboard()

if __name__ == "__main__":
    # Check for required dependencies
    try:
        from kivy.garden.matplotlib import backend_kivyagg
    except ImportError:
        print("Installing kivy-garden.matplotlib...")
        import os
        os.system("pip install kivy-garden matplotlib")
        os.system("garden install matplotlib")
    
    AgriApp().run()