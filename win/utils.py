import pickle
def write_pickle(d, path):
    with open(path, 'wb') as fp:
        pickle.dump(d, fp)

def read_pickle(path):
    with open(path, 'rb') as fp:
        return pickle.load(fp)
