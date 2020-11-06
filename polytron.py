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
        self.is_playing = False 
        self.adsr = [0,0,0,0]

    def set_adsr(adsr):
    	self.adsr = adsr 

    def get_is_playing(self):
    	return self.is_playing

    def _attack(self):
        logging.info("setting attack for %ds", self.adsr[0])
		time.sleep(self.adsr[0])

    def _release(self):
        logging.info("releasing for  %ds", self.adsr[3])

    def play(self,a,d,s,r):
        self.is_playing=True
        x=threading.Thread(target=self._attack,args=(a,d,s,r,))
        x.start()

    def stop(self,t):
        self.is_playing=False
        x=threading.Thread(target=self._release,args=(t,r))
        x.start()

# d=Sample('sample1')
# d2=Sample('sample2')
# print(d.kind)
# d.justsleep(1.1)
# d2.justsleep(1.3)
# print("test")
# time.sleep(3)
# print("done")