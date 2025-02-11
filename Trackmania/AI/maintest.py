from test import run
from test2 import test
import threading

t = test()

thread = threading.Thread(target=t.run)

thread.start()

print("Hi!")
run(t)