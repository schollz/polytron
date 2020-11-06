import logging
import threading
import time
logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO,datefmt="%H:%M:%S")


# import smbus
# import time

# # Get I2C bus
# bus = smbus.SMBus(1)

# # AD5667 address, 0x0E(14)
# # Select DAC and input register, 0x1F(31)
# #		0x8000(32768)
# voltage=3.33
# n = int(voltage/5.0*65535)
# lo = n & 0x00ff
# hi = n  >> 8
# data=[hi,lo]
# data = [0x80, 0x00]
# bus.write_i2c_block_data(0x0E, 0x1F, data)

# time.sleep(0.5)

# # Convert the data
# voltage = ((data[0] * 256 + data[1]) / 65536.0) * 5.0

# # Output data to screen
# print "Voltage : %.2f V" %voltage


class Voice:
    kind = 'canine'         # class variable shared by all instances

    def __init__(self, name):
        self.name = name    # instance variable unique to each instance
        self.last_played = time.time()

    def _justsleep(self,t):
        print("running")
        logging.info("Thread %s: starting", self.name)
        time.sleep(t)
        logging.info("Thread %s: finishing", self.name)

    def justsleep(self,t):
        x=threading.Thread(target=self._justsleep,args=(t,))
        x.start()

d=Sample('sample1')
d2=Sample('sample2')
print(d.kind)
d.justsleep(1.1)
d2.justsleep(1.3)
print("test")
time.sleep(3)
print("done")