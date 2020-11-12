import time
import sys
import os
import logging
import threading
import random
import math
import json
from subprocess import Popen, PIPE, STDOUT
from signal import signal, SIGINT

sem = threading.Semaphore()

import click
import numpy as np
from loguru import logger
import mido
from mido.ports import MultiPort
import smbus
import termplotlib as tpl

logger.remove()
logger.add(sys.stderr, level="ERROR")
# Get I2C bus
bus = smbus.SMBus(1)

VOLTAGE_VDD = 5.0
CHANNEL_PITCH = [32, 34, 36]
CHANNEL_CUTOFF = [33, 35, 37]
CHANNEL_NAMES = {}
for _, v in enumerate(CHANNEL_CUTOFF):
    CHANNEL_NAMES[v] = "cutoff {}".format(v)
for _, v in enumerate(CHANNEL_PITCH):
    CHANNEL_NAMES[v] = "pitch {}".format(v)

def freq2voltage(freq,fitting):
    return fitting[0] * math.log(freq) + fitting[1]

def midi2freq(midi_number):
    a = 440  # frequency of A (coomon value is 440Hz)
    return (a / 32) * (2 ** ((midi_number - 9) / 12))

def note2voltage(note,fitting):
    return freq2voltage(midi2freq(note),fitting)

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
    # 0x56 determined by sudo i2cdetect -y 1
    bus.write_i2c_block_data(0x56, channel, data)
    logger.debug("{} set to {:2.2f}", channel, voltage)

#
# frequency analysis
#

def get_frequency_analysis():
    cmd = "arecord -d 1 -f cd -t wav -D sysdefault:CARD=1 /tmp/1s.wav"
    p = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True)
    output = p.stdout.read()
    if b"Recording WAVE" not in output:
        raise output
    # cmd = "sox /tmp/1s.wav -n stat -freq"
    cmd = "aubio pitch -m schmitt -H 1024 /tmp/1s.wav"
    p = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True)
    output = p.stdout.read()
    with open("/tmp/1s.dat", "wb") as f:
        f.write(output)

    freq = analyze_aubio()
    return freq

def analyze_aubio():
    gathered_freqs = []
    with open("/tmp/1s.dat", "r") as f:
        linenum = 0
        for line in f:
            linenum += 1
            if linenum < 5:
                continue
            s = line.split()
            if len(s) != 2:
                continue
            freq = float(s[1])
            if freq > 100:
                gathered_freqs.append(freq)
    if len(gathered_freqs) == 0:
        return -1
    avg = np.median(gathered_freqs)
    return avg

#
# plotting 
#

def plot_points(voltage_to_frequency):
    x = []
    y0 = []
    for k in voltage_to_frequency:
        x.append(float(k))
        y0.append(voltage_to_frequency[k])
    fig = tpl.figure()
    print("\n")
    fig.plot(
        x,
        y0,
        plot_command="plot '-' w points",
        width=50,
        height=20,
        xlabel="voltage (v)",
        title="frequency (hz) vs voltage",
    )
    fig.show()
    print("\n")


class Envelope:
    """
    Some monotrons may share an envelope.
    """

    max_seconds = 10
    max_voltage = 4.0
    min_voltage = 0.0
    steps = 50

    def __init__(self, voice):
        self.voice = voice
        self.last_played = time.time()
        self.is_attacking = False
        self.is_releasing = False
        self.a = 0.5
        self.d = 0.05
        self.s = 0.5
        self.r = 0.4
        self.a= 0.1
        self.r = 0.3
        self.peak = 1.0
        self.value = 0
        self.set_adsr(self.peak,self.a,self.d,self.s,self.r)

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
        self.a = a * self.max_seconds
        self.d = d * self.max_seconds
        self.s = s * (self.max_voltage - self.min_voltage) + self.min_voltage
        self.r = r * self.max_seconds

    def on(self):
        set_voltage(CHANNEL_CUTOFF[self.voice], 5)

    def off(self):
        set_voltage(CHANNEL_CUTOFF[self.voice], 0)

    def attack(self, voice, velocity):
        if self.is_attacking:
            return
        self.is_attacking = True
        self.is_releasing = False
        x = threading.Thread(target=self._attack, args=(voice, velocity,))
        x.start()

    def _attack(self, voice, velocity):
        # attack
        logger.debug("attacking for {}s", self.a)
        step = (self.peak*velocity - self.value) / self.steps
        a = self.a - (velocity)*self.max_seconds # shorten attack if played harder
        if a < 0:
            a = 0.1
        for i in range(self.steps):
            if not self.is_attacking:
                return
            self._increment_cutoff(step)
            time.sleep(a / self.steps)
        set_voltage(CHANNEL_CUTOFF[self.voice], self.peak*velocity )

        # decay
        logger.debug("decaying for {}s", self.d)
        step = (self.s - self.value) / self.steps
        for i in range(self.steps):
            if not self.is_attacking:
                return
            self._increment_cutoff(step)
            time.sleep(self.d / self.steps)
        set_voltage(CHANNEL_CUTOFF[self.voice], self.s)
        self.is_attacking = False

    def release(self, voice):
        if self.is_releasing:
            return
        self.is_releasing = True
        self.is_attacking = False
        x = threading.Thread(target=self._release, args=(voice,))
        x.start()

    def _release(self, voice):
        # release
        logger.debug("releasing for {}s", self.r)
        step = (self.min_voltage - self.value) / self.steps
        for i in range(self.steps):
            if not self.is_releasing:
                return
            self._increment_cutoff(step)
            time.sleep(self.r / self.steps)
        set_voltage(CHANNEL_CUTOFF[self.voice], self.min_voltage)
        self.is_releasing = False


class Voices:
    """
    Voices keeps track of the tuning and availability
    of each voice.
    There should be one voice object passed into each
    of the keyboards.
    Voices are indexed starting at 0.
    """

    def __init__(self, max_voices, voice_envelope_mapping=[]):
        self.max_voices = max_voices
        self.voices = [0] * max_voices
        self.notes_used = {}
        self.voices_used = {}
        self.last_note = {}
        self.tuning = [{}] * max_voices
        self.envelope = []
        self.portamento=[0]*max_voices
        self.mbs = [] # stores curve fitting for frequencies
        self.v2e = voice_envelope_mapping
        if len(self.v2e) != self.max_voices:
            self.v2e = list(range(self.max_voices))
        for i in range(self.max_voices):
            self.envelope.append(Envelope(self.v2e[i]))

    def set_portamento(self,voice,portamento):
        self.portamento[voice]=portamento

    def set_adsr(self, voice, a, d, s, r):
        self.envelope[v2e[voice]].set_adsr(a, d, s, r)

    def tune(self,specific_voice=-1):
        for voice in range(self.max_voices):
            if specific_voice > 0 and specific_voice != voice:
                continue
            self.off()
            voltage_to_frequency = {}
            previous_freq = 0
            for voltage in range(260,380,5):
                voltage = float(voltage)/100.0
                self.solo_voltage(voice,voltage)
                time.sleep(1)
                freq = get_frequency_analysis()
                if freq < previous_freq:
                    continue
                voltage_to_frequency[voltage]=freq 
                previous_freq = freq 
                os.system("clear")
                print("voice {}".format(voice))
                plot_points(voltage_to_frequency)
            with open("voltage_to_frequency{}.json".format(voice), "w") as f:
                f.write(json.dumps(voltage_to_frequency))
        self.off()
        self.load_tuning()

    def load_tuning(self):
        self.mbs = []
        for voice in range(self.max_voices):
            voltage_to_frequency = json.load(open("voltage_to_frequency{}.json".format(voice), "rb"))
            x = []
            y = []
            y0 = []
            for k in voltage_to_frequency:
                x.append(float(k))
                y0.append(voltage_to_frequency[k])
                y.append(math.log(voltage_to_frequency[k]))
            mb = np.polyfit(y, x, 1)
            fig = tpl.figure()
            print("\n")
            fig.plot(
                x,
                y0,
                plot_command="plot '-' w points",
                width=60,
                height=22,
                xlabel="voltage (v)",
                title="frequency (hz) vs voltage",
                label="freq = exp((volts{:+2.2f})/{:2.2f})   ".format(mb[1], mb[0]),
            )
            fig.show()
            print("\n")
            time.sleep(0.1)
            self.mbs.append(mb)


    def solo_voltage(self,voice,voltage):
        for i in range(self.max_voices):
            self.envelope[self.v2e[i]].off()
        self.envelope[self.v2e[voice]].on()
        set_voltage(CHANNEL_PITCH[voice],voltage)

    def solo(self, voice):
        """
        Turns up cutoff for voice
        Removes other voices
        """
        for i in range(self.max_voices):
            if i == voice:
                set_voltage(CHANNEL_GATE[voice], GATE_ON)
            else:
                set_voltage(CHANNEL_GATE[voice], GATE_OFF)
            self.envelope[self.v2e[i]].off()
        self.envelope[self.v2e[voice]].on()

    def off(self):
        for i in range(self.max_voices):
            set_voltage(CHANNEL_CUTOFF[i],0)
            set_voltage(CHANNEL_PITCH[i],0)


    def acquire_voice(self):
        for voice in range(self.max_voices):
            if voice not in self.voices_used:
                return voice 
        
        # find oldest voice
        oldest = 0
        voice = 1
        for i, v in enumerate(self.voices):
            if time.time() - v > oldest:
                oldest = time.time() - v
                voice = i
        return voice 

    def play(self, note, velocity):
        if note in self.notes_used:
            return
        sem.acquire()
        voice = self.acquire_voice()
        self.voices[voice] = time.time()

        # remove voice if it was acquired
        if voice in self.voices_used:
            note_to_remove = self.voices_used[voice]
            logger.debug("removing voice {} playing {}", voice, note_to_remove)
            del self.voices_used[voice]
            del self.notes_used[note_to_remove]
        delete_note = []

        self.notes_used[note] = voice
        self.voices_used[voice] = note
        sem.release()
        logger.info("playing {} on voice {}", note, voice)

        # do legato
        if voice in self.last_note and self.portamento[voice] > 0:
            x = threading.Thread(target=self.play_legato, args=(voice,self.last_note[voice],note,))
            x.start()
        else:
            set_voltage(CHANNEL_PITCH[voice], note2voltage(note,self.mbs[voice]))
        self.envelope[self.v2e[voice]].attack(voice,velocity)
        self.last_note[voice]=note 

    def play_detuned(self,voice,note):
        set_voltage(CHANNEL_PITCH[voice], note2voltage(note,self.mbs[voice]))
        freq = midi2freq(note)
        min_freq = freq*(1-0.005)
        max_freq = freq*(1+0.005)
        for i in range(30):
            if voice not in self.voices_used or self.voices_used[voice] != note: # note is canceled
                return
            set_voltage(CHANNEL_PITCH[voice],freq2voltage(random.random()*(max_freq-min_freq)+min_freq,self.mbs[voice]))
            time.sleep(0.05)

    def play_legato(self,voice,note1,note2):
        set_voltage(CHANNEL_PITCH[voice], note2voltage(note1,self.mbs[voice]))
        freq1 = midi2freq(note1)
        freq2 = midi2freq(note2)
        steps = int(self.portamento[voice]/0.025)
        for i in range(steps):
            if voice not in self.voices_used or self.voices_used[voice] != note2: # note is canceled
                return
            time.sleep(0.025)
            set_voltage(CHANNEL_PITCH[voice],freq2voltage( (freq2-freq1)*(i/steps)+freq1,self.mbs[voice] ) )
        set_voltage(CHANNEL_PITCH[voice],freq2voltage(freq2,self.mbs[voice]))


    def stop(self, note):
        if note in self.notes_used:
            voice = self.notes_used[note]
            logger.debug("stopping {} on voice {}", note, voice)
            self.envelope[self.v2e[voice]].release(voice)
            sem.acquire()
            del self.voices_used[voice]
            del self.notes_used[note]
            sem.release()


class Keyboard:
    def __init__(self, name, num_voices):
        self.num_voices = num_voices
        self.voices = Voices(self.num_voices)
        self.name = name
        name = name.split()
        if len(name) > 2:
            name = " ".join(name[:2])
        else:
            name = " ".join(name)
        name = name.lower()
        name = name.replace(":", "")
        self.id = name

    def tune(self,specific_voice=-1):
        self.voices.tune(specific_voice)

    def load_tuning(self):
        self.voices.load_tuning()

    def listen(self):
        for name in mido.get_output_names():
            t = threading.Thread(target=self._listen, args=(name,))
            t.daemon = True
            t.start()

    def play(self,note,velocity):
        self.voices.play(note,velocity)

    def stop(self,note):
        self.voices.stop(note)

    def _listen(self,name):
        with mido.open_input(name) as inport:
            for msg in inport:
                if msg.type == "note_on":
                    note_name = midi2str(msg.note)
                    logger.debug(
                        f"[{name}] {note_name} {msg.type} {msg.note} {msg.velocity}"
                    )
                    self.voices.play(msg.note,msg.velocity/127)
                elif msg.type == "note_off":
                    note_name = midi2str(msg.note)
                    logger.debug(
                        f"[{name}] {note_name} {msg.type} {msg.note} {msg.velocity}"
                    )
                    self.voices.stop(msg.note)


for i in range(32,38):
    set_voltage(i,0)
keys = Keyboard("monotron",3)
# keys.tune()
keys.load_tuning()
keys.listen()
time.sleep(60000)
# keys.play(60)
# keys.play(61)
# keys.play(62)
# time.sleep(3)
# keys.stop(60)
# keys.stop(61)
# keys.stop(62)
# time.sleep(3)
