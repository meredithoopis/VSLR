import numpy as np

def _euclid_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        
def average_pose_to_hand_distance(palms, hands):
    distances = []
    
    for palm in palms:
        for hand in hands:
            distances.append(_euclid_distance(palm.x,
                                              palm.y,
                                              hand.x,
                                              hand.y))
    
    return np.mean(distances)