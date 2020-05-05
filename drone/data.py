from mat73 import loadmat

if __name__ == "__main__":
    mat = loadmat('dane.mat')
    #print(mat)
    print(mat.items())