import torch
TRAIN_IMAGES_PATH = "mnist/train-images.idx3-ubyte"
TRAIN_LABELS_PATH = "mnist/train-labels.idx1-ubyte"


def read_file():

    fd = open(TRAIN_IMAGES_PATH, 'rb')
    #  read header, get data size first
    dataType = int.from_bytes(fd.read(3), "big")
    if dataType < 10:
        chunksize = 1
    elif dataType < 12:
        chunksize = 2
    else:
        chunksize = 3

    #  then dimension count
    num_dims = int.from_bytes(fd.read(1), "big")

    #  read the info about each dimension's size
    dimensions = []
    for i in range(num_dims):
        dimensions.append(int.from_bytes(fd.read(4), "big"))

    images = torch.empty(dimensions[0], 1, *dimensions[1:])



if __name__ == "__main__":
    read_header()
