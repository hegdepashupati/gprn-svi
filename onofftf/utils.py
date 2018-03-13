import time
import numpy as np

def printtime(start):
    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    return str("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

class modelmanager:
    def __init__(self,saver,sess,path):
        self.sess = sess
        self.saver = saver
        self.path = path

    def save(self):
        self.saver.save(self.sess,self.path)
        print("model saved in : "+self.path)

    def load(self):
        self.saver.restore(self.sess,self.path)
        print("model loaded from : "+self.path)
