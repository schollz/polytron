import logging
import threading

sem = threading.Semaphore()

from loguru import logger
import mido
from mido.ports import MultiPort


# import smbus
# import time

# # Get I2C bus
# bus = smbus.SMBus(1)

# # AD5667 address, 0x0E(14)
# # Select DAC and input register, 0x1F(31)
# #     0x8000(32768)
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
VOLTAGE_VDD = 5.0
CHANNEL_GATE = [0x22, 0x23, 0x24, 0x25, 0x26]
CHANNEL_PITCH = [0x22, 0x23, 0x24, 0x25, 0x26]
CHANNEL_CUTOFF = [0x21, 0x22]


def midi2freq(midi_number):
    a = 440  # frequency of A (coomon value is 440Hz)
    return (a / 32) * (2 ** ((midi_number - 9) / 12))


def midi2str(midi_number, sharp=True):
    """
    Given a MIDI pitch number, returns its note string name (e.g. "C3").
    """
    MIDI_A4 = 69
    num = midi_number - (MIDI_A4 - 4 * 12 - 9)
    note = (num + 0.5) % 12 - 0.5
    rnote = int(round(note))
    error = note - rnote
    octave = str(int(round((num - note) / 12.0)))
    if sharp:
        names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    else:
        names = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
    names = names[rnote] + octave
    if abs(error) < 1e-4:
        return names
    else:
        err_sig = "+" if error > 0 else "-"
        err_str = err_sig + str(round(100 * abs(error), 2)) + "%"
        return names + err_str


def set_voltage(channel, voltage):
    sem.acquire()
    n = int(voltage / VOLTAGE_VDD * 65535)
    lo = n & 0x00FF
    hi = n >> 8
    data = [hi, lo]
    sem.release()
    # bus.write_i2c_block_data(0x0E, channel, data)


class Envelope:
    max_seconds = 10
    max_voltage = 5.0
    min_voltage = 0.0
    steps = 10

    def __init__(self, voice):
        self.voice = voice
        self.last_played = time.time()
        self.is_attacking = False
        self.is_releasing = False
        self.a = 0
        self.d = 0
        self.s = 0
        self.r = 0
        self.peak = 1.0
        self.value = 0

    def _increment_cutoff(self, val):
        self.value = self.value + val
        if self.value > self.max_voltage:
            self.value = self.max_voltage
        if self.value < self.min_voltage:
            self.value = self.min_voltage
        set_voltage(CHANNEL_CUTOFF[self.voice], self.value)

    def set_adsr(self, peak, a, d, s, r):
        # all input values are between 0 and 1
        self.peak = peak * (self.max_voltage - self.min_voltage) + self.min_voltage
        self.a = a * max_seconds
        self.d = d * max_seconds
        self.s = s * (self.max_voltage - self.min_voltage) + self.min_voltage
        self.r = r * max_seconds

    def attack(self):
        if self.is_attacking:
            return
        self.is_attacking = True
        self.is_releasing = False
        x = threading.Thread(target=self._attack, args=(a, d, s, r,))
        x.start()

    def _attack(self):
        logger.info("setting attack for %ds", self.a)

        # attack
        step = (self.peak - self.value) / steps
        for i in range(self.steps):
            if not self.is_attacking:
                return
            self._increment_cutoff(step)
            time.sleep(self.a / steps)
        set_voltage(CHANNEL_CUTOFF[self.voice], self.peak)

        # decay
        step = self.s - self.value
        for i in range(self.steps):
            if not self.is_attacking:
                return
            self._increment_cutoff(step)
            time.sleep(self.d / steps)
        set_voltage(CHANNEL_CUTOFF[self.voice], self.s)
        self.is_attacking = False

    def release(self, t):
        if self.is_releasing:
            return
        self.is_releasing = True
        self.is_attacking = False
        x = threading.Thread(target=self._release, args=(t, r))
        x.start()

    def _release(self):
        # release
        step = self.min_voltage - self.value
        for i in range(self.steps):
            if not self.is_releasing:
                return
            self._increment_cutoff(step)
            time.sleep(self.d / steps)
        set_voltage(CHANNEL_CUTOFF[self.voice], min_voltage)
        self.is_releasing = False


class Voices:
    """ 
    Voices keeps track of the tuning and availability
    of each voice.
    There should be one voice object passed into each
    of the keyboards.
    """

    def __init__(self, max_voices, voice_envelope_mapping):
        self.max_voices = max_voices
        self.voices = [0] * max_voices
        self.notes_on = {}
        self.tuning = [{}] * max_voices
        self.envelope = []
        # voice_envelope_mapping = [0,1,1] => maps voice 0 to envelope 0 and voice 1 and 2 to envelope 1
        self.voice_envelope_mapping = voice_envelope_mapping
        for i in range(max(voice_envelope_mapping) + 1):
            self.envelope.append(Envelope(voice_envelope_mapping[i]))

    def set_adsr(self, voice, a, d, s, r):
        self.envelope[voice_envelope_mapping[voice]].set_adsr(a, d, s, r)

    def tune(self):
        # TODO do tuning of each voice
        pass

    def play(self, note):
        if note in notes_on:
            return
        sem.acquire()
        # find oldest voice
        oldest = 0
        voice = 1
        for i, v in enumerate(self.voices):
            if time.time() - v > oldest:
                oldest = time.time() - v
                voice = i
        self.voices[voice] = time.time()
        self.notes_on[note] = voice
        sem.release()
        logger.debug("playing {} on voice {}", note, voice)
        set_voltage(CHANNEL_GATE[voice], 0)
        # TODO: compute voltage from tuning of voice
        set_voltage(CHANNEL_PITCH[voice], 0)
        self.envelope[self.voice_envelope_mapping[voice]].attack()

    def stop(self, note):
        if note in notes_on:
            voice = notes_on[note]
            logger.debug("stopping {} on voice {}", note, voice)
            set_voltage(CHANNEL_GATE[voice], 0)
            self.envelope[voice_envelope_mapping[voice]].release()
            sem.acquire()
            del notes_on[note]
            sem.release()


class Keyboard:
    def __init__(self, name, voices):
        self.voices = voices
        self.name = name
        name = name.split()
        if len(name) > 2:
            name = " ".join(name[:2])
        else:
            name = " ".join(name)
        name = name.lower()
        name = name.replace(":", "")
        self.id = name

    def listen(self):
        t = threading.Thread(target=self._listen, args=())
        t.daemon = True
        t.start()

    def _listen(self):
        with mido.open_input(name) as inport:
            for msg in inport:
                if msg.type == "note_on":
                    note_name = midi2str(msg.note)
                    logger.info(
                        f"[{name}] {note_name} {msg.type} {msg.note} {msg.velocity}"
                    )
                    self.voices.play(msg.note)


class A:
    def __init__(self):
        self.a = 1

    def increment(self):
        self.a = self.a + 1
        print(self.a)


class B:
    def __init__(self, a):
        self.a = a

    def increment(self):
        self.a.increment()


logger.debug("running")
a = A()
a.increment()
b = B(a)
c = B(a)
b.increment()
a.increment()
b.increment()
c.increment()
# d=Sample('sample1')
# d2=Sample('sample2')
# print(d.kind)
# d.justsleep(1.1)
# d2.justsleep(1.3)
# print("test")
# time.sleep(3)
# print("done")
