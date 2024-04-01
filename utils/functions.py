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

def normalize_ignore_nan(ar):
    arr = np.array(ar)
    value = []
    index = []
    
    for i, a in enumerate(arr):
        if not np.isnan(a):
            value.append(a)
            index.append(i)
    
    value = np.array(value)
    mean = np.mean(value)
    std = np.std(value)
    
    value = (value - mean)/std
    
    arr[index] = value

    return arr

if __name__ == "__main__":
    arr = np.array([1,2,np.nan,2,1])
    print(arr)
    print(normalize_ignore_nan(arr))