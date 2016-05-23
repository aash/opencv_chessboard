

from os import system
from glob import glob

for i in glob("data/*.png"):
    system('python chessboard.py %s -s' % i)