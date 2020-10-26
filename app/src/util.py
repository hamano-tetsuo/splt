import os
import datetime
import shutil
import json
from enum import Enum


def make_hardlink(src_file, dist_dir):
    os.link(src_file, dist_dir + "/" + os.path.basename(src_file))


def mprof_timestamp(msg):
    try:
        with profile.timestamp(msg):
            #print(f"mprof_timestamp msg={msg}")
            #time.sleep(1)
            pass
    except NameError:
        pass


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, o):  # pylint: disable=E0202
        if isinstance(o, Enum):
            return f"{o.__class__.__name__}.{o.name}"
        try:
            json.dumps(o)
        except TypeError:
            return str(o)
            
        return super(CustomJSONEncoder, self).default(o)


def str2bool(s):
    if s == "True":
        return True
    elif s == "False":
        return False
    else:
        assert False, f"s = {s}"


def dump_json(dat, path):
    json.dump(dat, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=4, sort_keys=False, separators=(',', ': '), cls=CustomJSONEncoder)


def load_json(path):
    return json.load(open(path, "r", encoding="utf-8"))


def get_time_str():
    return datetime.datetime.now().strftime('%y%m%d_%H%M%S_%f')


def trash(src):
    if not os.path.exists(src):
        print(f"not exists {src}")
        return
    srcbase = os.path.basename(src)
    timestr = get_time_str()
    dist =  f"./trash/{srcbase}_{timestr}"
    if os.path.isdir(src):
        #src += "/"
        dist += "/"
    if not os.path.exists("./trash"):
        os.mkdir("./trash")
    print(f"move {src}⇒{dist}")
    shutil.move(src, dist)