import numpy as np, pickle as pkl, matplotlib.pyplot as plt



def main():
    ex5('aba')

    ex6('aba')

    ex7()

    plt.show()


def ex5(text_to_align):
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

    mapping = {
            'a':0,
            'b':1,
            '^':2
        }
    
    prob, forward_probs = forwardPass(pred,text_to_align,mapping)

    print(f'The probability of the sequence {text_to_align} is: {prob}')
    plotForwrdMat(forward_probs, text_to_align)



def ex6(text_to_align):
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

    mapping = {
            'a':0,
            'b':1,
            '^':2
        }
    
    prob, path, forward_probs = forwardPass_forceAlign(pred,text_to_align,mapping)

    print(f'The probability of the sequence {text_to_align} is: {prob}')
    print(f'The taken path is: {path}')
    plotForwrdMat(forward_probs, text_to_align, path=path, isq6=True)

def ex7():
    data = pkl.load(open('./force_align.pkl', 'rb'))
    mapping = data['label_mapping']
    mapping = {value: key for key, value in mapping.items()}
    pred = data['acoustic_model_out_probs']
    GT = data['text_to_align']

    prob, path, forward_probs = forwardPass_forceAlign(pred,GT,mapping)
    
    print(f'The probability of the sequence {GT} is: {prob}')
    print(f'The taken path is: {path}')
    plotForwrdMat(forward_probs, GT, isq7=True)

def plotForwrdMat(alpha, GT, path=None, isq6=False, isq7=False):
    text = addBlanks(GT)
    text = [c for c in text]
    
    fig = plt.figure()
    ax = fig.add_subplot()
    cax = ax.matshow(alpha.T, cmap='viridis',interpolation='nearest')
    fig.colorbar(cax)

    ax.tick_params(axis='x', bottom=True, top=False, labelbottom=True, labeltop=False)

    if (not isq7):
        x_labels = range(alpha.shape[0])
        ax.set_yticklabels(['']+text)
        ax.set_xticks(x_labels)
        plt.xticks(ticks=x_labels, labels=x_labels)
        plt.title(f'CTC forward matrix for {GT}')

    ax.set_xlabel("Time")
    ax.set_ylabel("Character")

    if (isq6):
        ax_top = ax.secondary_xaxis("top")
        ax_top.set_xlim(ax.get_xlim())  # Match the limits of the bottom x-axis
        ax_top.tick_params(axis='x', bottom=False, top=True, labelbottom=False, labeltop=True)
        top_x_labels = path  # Custom labels for the top x-axis
        ax_top.set_xticks(x_labels)
        ax_top.set_xticklabels(top_x_labels)
        ax_top.set_xlabel("Tekn path")

    elif (isq7): plt.title(f'CTC forward matrix for the given data')


def forwardPass(pred, GT, translation):
    # pred - a T x n size matrix, where T is time and n is the vocabulary size
    # GT - a string, the ground truth audio
    #translation - a dictionary mapping between characters to indices, with entries of the form 'a':0.

    truth = addBlanks(GT)

    T = len(pred)
    S = len(truth)
    
    alpha = initAlphaMatrix(pred, truth, T, S, translation)

    for t in range(1, T):
        for s in range(S):
            alpha[t, s] = alpha[t - 1, s]
            if s > 0:
                alpha[t, s] += alpha[t - 1, s - 1]
            if s > 1 and truth[s] != truth[s - 2] and truth[s] != '^':
                alpha[t, s] += alpha[t - 1, s - 2]

            alpha[t, s] *= pred[t, translation[truth[s]]]
            
    res = alpha[T-1][S-1] + alpha[T-1][S-2]


    return res, alpha


def forwardPass_forceAlign(pred, GT, translation):
    # pred - a T x n size matrix, where T is time and n is the vocabulary size
    # GT - a string, the ground truth audio
    #translation - a dictionary mapping between characters to indices, with entries of the form 'a':0.

    truth = addBlanks(GT)

    T = len(pred)
    S = len(truth)
    
    alpha = initAlphaMatrix(pred, truth, T, S, translation)
    backPointers = np.zeros((T, S))

    for t in range(1, T):
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

    return res,path, alpha


def addBlanks(s):
    res = "^"
    for c in s:
        res += c
        res += "^"

    return res

def initAlphaMatrix(pred,s, T,S, translation):
    alpha = np.zeros((T, S))
    alpha[0][0] = pred[0][translation[s[0]]]
    alpha[0][1] = pred[0][translation[s[1]]]

    return alpha


def collapse(s):
    parts = s.split('^')
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