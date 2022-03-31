from sys import path
from os import path as ospath
path.append(ospath.join(ospath.dirname(__file__), ".."))
import cv2
import numpy as np
import matplotlib.pyplot as plt
from random import choice

class Counter():
    def __init__(self):
        self.colours = {
            "Yellow": 0,
            "Green": 1,
            "Gray": 2,
            "Blue": 3,
            "Black": 4,
            "Orange": 5,
            "White": 6,
            "Red": 7,
            "Pink": 8
        }
        self.votes = np.zeros(9, dtype=np.float32)
    
    def addVote(self, colour, confidence=1):
        idx = self.colours[colour]
        self.votes[idx] += confidence

    def getWinner(self):
        return list(self.colours.keys())[self.votes.argmax()]
    
    def getWinningPercentage(self, total_votes=None):
        if total_votes is None:
            total_votes = self.getTotalVotes()
        winner = self.getWinner()
        winning_num_votes = self.votes[self.colours[winner]]
        return winning_num_votes/total_votes

    def getTotalVotes(self):
        return np.sum(self.votes)

    def printBreakdown(self, total_votes=None):
        if total_votes is None:
            total_votes = self.getTotalVotes()
        for i, col in enumerate(self.colours):
            print(f"{col}: {self.votes[i]}/{total_votes}")
            
    def __add__(self, other):
        total = self.votes + other.votes
        output_counter = Counter()
        output_counter.votes = total
        return output_counter

if __name__ == "__main__":
    c = Counter()
    for i in range(100):
        col = choice(list(c.colours.keys()))
        c.addVote(col)
    print(c.getWinner())
    c.printBreakdown()