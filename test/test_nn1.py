import unittest
import sys

sys.path.append("../")
from src.pworld import *
from src.pnn import *

nround = lambda x : np.round(x, 2)

def comp_loss(predict: np.array, observ: np.array):
    """Compute the MSE between prediction and observations
    Args:
        predict: predictions, [N, 2, 2]
        observ: observations, [N, 4]
    """
    a, b = predict.flatten(), observ.flatten()
    n = len(a)
    return np.sum((a-b)**2) / n

def test_case(pa: np.array, seq:int = 8,
              file:str = None,
              verbose:bool = False):
    """Test 1 case
    Args:
        pa: the input data, [num, 15], (r, v, a, ct, bbox)
        seq: the number of input sequence
        file: if not None, then write the tracjectories to
            this file
        verbose: whether to print detailed info
    """
    num = len(pa)
    m = MotionSolver()
    pose = None
    predictions = 4

    toFile = (file is not None)
    pb = np.zeros([num, 15])

    for i in range(0, num - seq - predictions + 1): 
        input = pa[i:(i+seq), 11:15].tolist()
        input = torch.tensor(input, requires_grad=False)

        pose, next_states, next_poses = m.solve(input, guess=pose)
        if verbose:
            print(i, "=> Estimated state =", 
                nround(pose.detach().numpy()))
            print("     Next 4 states: ",
                nround(next_states.detach().numpy()))
        
        j1, j2 = i+seq, i+seq+predictions
        n_states = next_states.detach().numpy()
        loss = comp_loss(n_states,  pa[j1:j2, 11:15])
        print(i, "->", loss)

        if toFile:
            pb[j1:j2, 0:9] = next_poses.detach().numpy()[:9]
            pb[j1:j2, 11:] = n_states.reshape([predictions, 4])
            pb[j1:j2, 9:11] =  (pb[j1:j2, 11:13] + pb[j1:j2, 13:15]) / 2 # center
    
    if toFile:
        cg = CaseGenerator()
        cg.config()
        cg.addObject("observation", r = (0,0,0), v = (0, 0, 0))
        cg.addObject("prediction", r = (0, 0, 0), v = (0, 0, 0))
        cg.write(file, [pa[seq:, :], pb[seq:, :]])



def test_nn1(file, testNum:int = None):
    # -- get data from this file
    pa = CaseGenerator.read(file)
    tsn = len(pa)
    num = len(pa[0])

    pa = arr(pa)    # [tsn, num, 15]

    # -- generate a list of test cases
    if (testNum is None) or testNum > tsn:
        testNum = int(tsn/2)
    cases = random.sample(range(0, tsn), testNum)

    for c in cases:
        print("testing case", c, "....")
        test_case(pa[c, :, :], seq = 8, file = "case" + str(c+1) + ".txt")

if __name__ == "__main__":
    file = "test2.txt"
    testNum = 1
    if len(sys.argv) >= 2:
        file = sys.argv[1]
    if len(sys.argv) >= 3:
        testNum = int(sys.argv[2])

    test_nn1(file, testNum = testNum)


    
