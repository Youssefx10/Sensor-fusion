import serial
import matplotlib.pyplot as plt
import time

PORT = 'COM4'  # or '/dev/ttyUSB0' on Linux
BAUD = 9600

# Open serial connection
ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2)  # Wait for the connection to stabilize

# Set up the plot
plt.ion()
fig, ax = plt.subplots()
sc = ax.scatter([], [], c='green', s=10)
ax.set_xlim(-200, 200)
ax.set_ylim(0, 200)
ax.set_aspect('equal')
ax.set_title("Radar-like Ultrasonic Map")
ax.set_xlabel("X (cm)")
ax.set_ylabel("Y (cm)")

x_data = []
y_data = []

try:
    while True:
        line = ser.readline().decode().strip()
        if not line:
            continue
        try:
            x_str, y_str = line.split(',')
            x = float(x_str)
            y = float(y_str)
            x_data.append(x)
            y_data.append(y)

            if len(x_data) > 100:  # limit points shown
                x_data.pop(0)
                y_data.pop(0)

            sc.set_offsets(list(zip(x_data, y_data)))
            plt.pause(0.001)

        except ValueError:
            # Ignore lines that aren't valid numbers
            continue

except KeyboardInterrupt:
    print("Exiting...")
    ser.close()
