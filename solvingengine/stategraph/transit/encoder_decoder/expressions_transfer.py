from copy import deepcopy
import re


# An expression tree node
class Et:
    # Constructor to create a node
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

