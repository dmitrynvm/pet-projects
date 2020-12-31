import numpy
import pandas
import os
import time
from fnmatch import fnmatch


prefixes = [
    'eyetracker_*.csv',
    'gsi_*.csv',
    'key_*.csv',
    'mkey_*.csv',
    'mxy_*.csv'
]


def nanmedian(series):
    output = 0
    for _, val in series.items():
        if not pandas.isnull(val):
            output += val
    top = output
    bot = max(series.size, 1)
    output = round(top / bot, 3)
    if not output:
        return numpy.nan
    return output


def nanconcate(series):
    output = '{}'
    for _, val in series.items():
        if not pandas.isnull(val):
            if len(val) > 2: # not '', '{}'
                if len(output) > 2:
                    output = output[:-1] + ',' + val[1:]
                else:
                    output = val
    return output


drivers = {
    'gaze_x': numpy.nanmedian,
    'gaze_y': numpy.nanmedian,
    'key': nanconcate,
    'mouse_dx': numpy.nanmedian,
    'mouse_dy': numpy.nanmedian,
    'mouse_key': nanconcate,
}


def match(name, patterns):
    return max([fnmatch(name, pattern) for pattern in patterns])


def load_one(path, patterns):
    frames = []
    for _, _, file_names in os.walk(path):
        for file_name in file_names:
            if match(file_name, patterns):
                file_path = os.path.join(path, file_name)
                frame = pandas.read_csv(file_path, error_bad_lines=False, index_col=0, parse_dates=True)
                frames.append(frame)
    return frames


def clean(frames):
    supported = drivers.keys()
    for frame in frames:
        given = frame.columns
        deleted = []
        for column in given:
            if column not in supported:
                deleted.append(column)
        frame.drop(columns=deleted, inumpylace=True)


def slice(frames):
    output = []
    head = max([frame.head(1).index[0] for frame in frames])
    tail = min([frame.tail(1).index[0] for frame in frames])
    for frame in frames:
        output.append(frame.iloc[:-1].truncate(head, tail, copy=False))
    return output


def merge(frames):
    return pandas.concat(frames, sort=True).sort_index()


def sample(frame, freq):
    timedelta = pandas.Timedelta(1. / freq, unit='s')
    reducers = {column: drivers[column] for column in frame.columns}
    return frame.resample(timedelta).agg(reducers)#.fillna(method='bfill', limit=1)


def split(frame, chunk):
    return [frame[i:i+chunk] for i in range(0, frame.shape[0], chunk)]


def save(frames, path):
    outputs = []
    os.makedirs(path, exist_ok=True)
    for idx, frame in enumerate(frames):
        file_path = os.path.join(path, 'data_{}.csv'.format(str(idx).zfill(2)))
        frame.to_csv(file_path)
        outputs.append(file_path)
    return outputs


def process_one(inumpyut_dir, output_dir, regexp, pause=None):
    print('PROCESS_ONE_STARTED')
    loaded = load_one(inumpyut_dir, regexp)
    if pause:
        time.sleep(pause)
    clean(loaded)
    if pause:
        time.sleep(pause)
    sliced = slice(loaded)
    if pause:
        time.sleep(pause)
    merged = merge(sliced)
    if pause:
        time.sleep(pause)
    sampled = sample(merged, 140)
    if pause:
        time.sleep(pause)
    splited = split(sampled, 1000)
    if pause:
        time.sleep(pause)
    save(splited, output_dir)
    print('PROCESS_ONE_FINISHED')
    return splited