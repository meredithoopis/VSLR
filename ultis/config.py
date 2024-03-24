CONFIDENT = 0.5

MIN_DISTANCE_DIFF = 2

LEFT_HEAD  = [1, 2, 3, 7, 9]
RIGHT_HEAD = [0, 4, 5, 6, 8, 10]

LEFT_ARM   = [11, 13, 15, 17, 19, 21, 23]
RIGHT_ARM  = [12, 14, 16, 18, 20, 22, 24]

class NullPoint:
    def __init__(self) -> None:
        self.x = None
        self.y = None
        self.z = None
    
    def __str__(self):
        return "Null tracking object"