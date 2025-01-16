import numpy as np
import librosa


translation = {
        'a':0,
        'b':1,
        '^':2
    }

def main():
    pred = np.zeros(shape=(5, 3), dtype=np.float32)
    pred[0][0] = 0.8
    pred[0][1] = 0.2
    pred[1][0] = 0.2
    pred[1][1] = 0.8
    pred[2][0] = 0.3
    pred[2][1] = 0.7
    pred[3][0] = 0.09
    pred[3][1] = 0.8
    pred[3][2] = 0.11
    pred[4][2] = 1.00


    print(forwardPass(pred,"ab"))


def forwardPass(pred, GT):
    # pred - a T x n size matrix, where T is time and n is the vocabulary size
    # GT - a string, the ground truth audio

    truth = addBlanks(GT)

    T = len(pred)
    S = len(truth)
    
    alpha = initAlphaMatrix(pred, truth, T, S)

    for t in range(1, T):
        for s in range(S):
            alpha[t, s] = alpha[t - 1, s]
            if s > 0:
                alpha[t, s] += alpha[t - 1, s - 1]
            if s > 1 and truth[s] != truth[s - 2] and truth[s] != '^':
                alpha[t, s] += alpha[t - 1, s - 2]

            alpha[t, s] *= pred[t, translation[truth[s]]]
            
    res = alpha[T-1][S-1] + alpha[T-1][S-2]


    return res


def forwardPass_forceAlign(pred, GT):
    # pred - a T x n size matrix, where T is time and n is the vocabulary size
    # GT - a string, the ground truth audio

    truth = addBlanks(GT)

    T = len(pred)
    S = len(truth)
    
    alpha = initAlphaMatrix(pred, truth, T, S)
    backPointers = np.zeros((T, S))

    for t in range(1, T):
        print(f'{alpha=}')
        for s in range(S):
            max_prob = alpha[t - 1, s]
            backpointer = s

            if s > 0 and alpha[t - 1, s - 1] > max_prob:
                max_prob = alpha[t - 1, s - 1]
                backpointer = s - 1

            if s > 1 and truth[s] != truth[s - 2] and alpha[t - 1, s - 2] > max_prob:
                max_prob = alpha[t - 1, s - 2]
                backpointer = s - 2

            alpha[t, s] = max_prob * pred[t, translation[truth[s]]]
            backPointers[t, s] = backpointer
            

    res = max(alpha[T-1][S-1], alpha[T-1][S-2])

    s = S-1 if res==alpha[T-1][S-1] else S-2
    path = ''
    for t in range(T):
        path += truth[s]
        s = int(backPointers[T-t-1, s])

    path = path[::-1]

    return res,path


def addBlanks(s):
    res = "^"
    for c in s:
        res += c
        res += "^"

    return res

def initAlphaMatrix(pred,s, T,S):
    alpha = np.zeros((T, S))
    alpha[0][0] = pred[0][translation[s[0]]]
    alpha[0][1] = pred[0][translation[s[1]]]

    return alpha


def collapse(s:str):
    parts = s.split('-')
    res = ''

    for part in parts:
        if part != '':
            res += part[0]

            for i in range(1,len(part)):
                if (part[i]!=part[i-1]):
                    res += part[i]

    return res
            

            

    







if __name__ == '__main__':
    main()