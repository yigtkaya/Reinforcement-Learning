import serial


while True:
    ser = serial.Serial('/dev/cu.usbserial-10', 9600, timeout=1)
    value= ser.readline()
    valueInString=str(value,'UTF-8')
    print(valueInString)