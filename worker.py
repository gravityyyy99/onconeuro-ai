"""Background worker: runs the SAME PyTorch + Grad-CAM code from model.py, but in its own process.

Why: importing PyTorch and Grad-CAM takes ~20 seconds and holds Python's lock the whole time. Inside the
website's process that froze every page. Here it happens next door, so the website stays instant.
Started by model.warmup(); talks to the website over stdin/stdout (length-prefixed pickles).
"""
import io
import os
import pickle
import struct
import sys

# keep the real stdout for our protocol; anything libraries print goes to stderr instead
_out = os.fdopen(os.dup(sys.stdout.fileno()), "wb")
sys.stdout = sys.stderr
_in = sys.stdin.buffer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def send(obj):
    b = pickle.dumps(obj, protocol=4)
    _out.write(struct.pack("<Q", len(b)) + b)
    _out.flush()


def recv():
    head = _in.read(8)
    if len(head) < 8:
        return None
    return pickle.loads(_in.read(struct.unpack("<Q", head)[0]))


if __name__ == "__main__":
    from PIL import Image

    import model

    model.get_model()          # the slow part: imports PyTorch, Grad-CAM, loads the weights
    send(("ready", None))
    while True:
        data = recv()
        if data is None:       # website closed the pipe
            break
        try:
            send(("ok", model.analyze_image(Image.open(io.BytesIO(data)))))
        except Exception as e:  # report instead of crashing the worker
            send(("err", repr(e)))
