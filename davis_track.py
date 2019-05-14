import numpy as np
import os
import traceback

class Trace(object):
    def __init__(self, params):
        self.times = []
        self.params = {}
        for p in params:
            self.params[p] = []
    def frame(self, t, **params):
        self.times.append(t)
        for p in self.params.keys():
            if p not in params:
                self.params[p].append(self.params[p][-1])
            else:
                self.params[p].append(params[p])

    def get(self, t, p):
        if t < self.times[0]:
            return None
        elif t > self.times[-1]:
            return None
        else:
            return np.interp([t], self.times, self.params[p])[0]


def load_trace(fn):
    if not os.path.isfile(fn):
        return None
    else:
        with open(fn) as f:
            code = f.read()
        locals = dict()
        globals = dict(Trace=Trace)
        try:
            exec(code, globals, locals)
        except:
            traceback.print_exc()
            return None
        for k, v in locals.items():
            if isinstance(v, Trace):
                return v
        else:
            return None

def extract_targets(filename, dt, t_start=None, t_end=None):
    trace = load_trace(filename+'.label')
    if t_end is None:        
        t_end = trace.times[-1] if trace is not None and len(trace.times)>0 else -1
    if t_start is None:
        t_start = 0
        
    times = []
    targets = []
    now = t_start
    while now < t_end:
        xx = trace.get(now, 'x')
        yy = trace.get(now, 'y')
        rr = trace.get(now, 'r')
        valid = 1 if xx is not None else 0
        if xx is None:
            xx = -1
        if yy is None:
            yy = -1
        if rr is None:
            rr = -1

        now += dt

        targets.append([xx, yy, rr, valid])
        times.append(now)

    targets = np.array(targets).reshape(-1, 4)
    times = np.array(times)
    
    return times, targets
        
    
    
def extract_images(filename,        # filename to load data from 
                   dt,              # time between images to create (seconds)
                   decay_time=0.1,  # spike decay time (seconds)
                   t_start=None,    # time to start generating images (seconds)
                   t_end=None       # time to end generating images (seconds)
                  ):
    packet_size = 8

    with open(filename, 'rb') as f:
        data = f.read()
    data = np.fromstring(data, np.uint8)

    # find x and y values for events
    y = ((data[1::packet_size].astype('uint16')<<8) + data[::packet_size]) >> 2
    x = ((data[3::packet_size].astype('uint16')<<8) + data[2::packet_size]) >> 1
    # get the polarity (+1 for on events, -1 for off events)
    p = np.where((data[::packet_size] & 0x02) == 0x02, 1, -1)
    v = np.where((data[::packet_size] & 0x01) == 0x01, 1, -1)
    # find the time stamp for each event, in seconds from the start of the file
    t = data[7::packet_size].astype(np.uint32)
    t = (t << 8) + data[6::packet_size]
    t = (t << 8) + data[5::packet_size]
    t = (t << 8) + data[4::packet_size]
    #t = t - t[0]
    t = t.astype(float) / 1000000   # convert microseconds to seconds

    if t_start is None:
        t_start = 0
    if t_end is None:
        t_end = t[-1]

    image = np.zeros((180, 240), dtype=float)

    images = []
    targets = []
    times = []

    event_index = 0   # for keeping track of where we are in the file
    if t_start > 0:
        event_index = np.searchsorted(t, t_start)

    now = t_start

    event_dt = dt

    while now < t_end:
        if event_dt != 0:
            decay_scale = 1-np.abs(event_dt)/(np.abs(event_dt)+decay_time)
            image *= decay_scale

        if event_dt > 0:
            count = np.searchsorted(t[event_index:], now + event_dt)
            s = slice(event_index, event_index+count)

            dts = event_dt-(t[s]-now)
            image[y[s], x[s]] += p[s] * (1-dts / (dts+decay_time))
            event_index += count

        now += event_dt

        images.append(image.copy())
        times.append(now)

    images = np.array(images)
    times = np.array(times)
    
    return times, images

