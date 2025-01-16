import numpy as np
import librosa


def main():
    print(collapse('---aaaabb-abbba--ba'))




def forwardPass(mat):
    pass



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