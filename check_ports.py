"""
Lists available serial ports and their device information.

Used to identify the COM port assigned to the AxiDraw controller.
"""


from serial.tools import list_ports

ports = list_ports.comports()

if not ports:
    print("No serial ports found.")

for port in ports:
    print(
        "Port:", port.device,
        "| Description:", port.description,
        "| Manufacturer:", port.manufacturer
    )