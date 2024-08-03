import numpy as np
from numpy import asarray
from PIL import Image,ImageDraw
import json
from tqdm import tqdm
import os.path
import math
import random
import cv2

class MNIST_READER:
    def readLInt(self,pos,size):
        self.labelsFile.seek(pos)
        return int.from_bytes(self.labelsFile.read(size), "big")
    def readIInt(self,pos,size):
        self.imagesFile.seek(pos)
        return int.from_bytes(self.imagesFile.read(size), "big")
    def readIByteArr(self,pos,size):
        self.imagesFile.seek(pos)
        return bytearray(self.imagesFile.read(size))
    
    def __init__(self, datadir, type="train"):
        self.datadir = datadir
        self.labelsFile = open(f"{datadir}{type}-labels.idx1-ubyte", "rb")
        self.imagesFile = open(f"{datadir}{type}-images.idx3-ubyte", "rb")
        self.samples = self.readLInt(4,4)

        assert(self.readLInt(0,4) == 2049)
        assert(self.readIInt(0,4) == 2051)
        assert(self.readLInt(4,4) == self.readIInt(4,4))
        self.rows = self.readIInt(8,4)
        self.cols = self.readIInt(12,4)
        
        print("*Reader Init Completed*")
        print(f"Image Size {self.rows}x{self.cols}")
        print(f"{self.samples} Samples Loaded")

    def __del__(self):
        self.labelsFile.close()
        self.imagesFile.close()

    def readImage(self,idx):
        label = self.readLInt(idx+8,1)
        rawByteData = self.readIByteArr((idx*(self.rows*self.cols))+16, self.rows*self.cols)
        image = np.reshape(rawByteData, (self.rows,self.cols))
        return (label, image)

class MNIST_VECTOR_DECIDE:
    @staticmethod
    def craftVectorObject(label,img):
        assert(len(img) == 28 and len(img[0]) == 28)
        vectorObject = {}
        vectorObject["label"] = label
        vectorObject["vector"] = []

        for y in range(0,28):
            for x in range(0,28):
                 vectorObject["vector"].append(img[x][y]/255)

        assert(len(img) == len(img[0]))
        vectorObject["vector"] =  np.clip(vectorObject["vector"],-1,1)
        vectorObject["vector"] = vectorObject["vector"].tolist()

        return vectorObject
    
def calcDistance(v1, v2):
    assert(len(v1) == len(v2))
    sum = 0
    for n in range(len(v1)):
        sum += (v1[n]-v2[n])**2
    return math.sqrt(sum)

def predict(vectorObject,vector = None,img = None, k = 1):
    assert(k < len(vectorObject))

    if vector == None:
        subject = MNIST_VECTOR_DECIDE.craftVectorObject(-1, img)
        vector = subject["vector"]

    neighbors = []

    for j in range(len(vectorObject)):
        x = vectorObject[str(j)]
        dist = calcDistance(vector, x["vector"])
        selected = (x["label"], x["vector"], dist)
        if len(neighbors) < k:
            neighbors.append(selected)
            continue
        for i in range(len(neighbors)):
            if neighbors[i][2] < selected[2]:
                continue
            neighbors[i], selected = selected, neighbors[i]
    return neighbors

def createFreqTable(prediction):
    freqTable = {}
    total = 0
    for i in prediction:
        total += 1
        if i[0] not in freqTable:
            freqTable[i[0]] = 1
            continue
        freqTable[i[0]] += 1
    return freqTable, total

def mostFrequent(freqTable):
    assert(len(freqTable) > 0)
    highest = None
    for i in freqTable:
        if highest == None:
            highest = i
            continue
        if freqTable[i] > freqTable[highest]:
            highest = i
    return highest
def sampleTest(vecOjb,testVector,size, k=1):
    tests = 0
    sucess = 0
    for i in tqdm(range(size)):
        idx = random.randrange(0,len(testVector))
        prediction = predict(vecOjb,vector=testVector[str(idx)]["vector"], k=k)
        freqTable, total = createFreqTable(prediction)
        highest= mostFrequent(freqTable)
        if str(highest) == str(testVector[str(idx)]["label"]):
            sucess += 1
        tests += 1
    return (sucess/tests)*100, sucess, tests

        
print("Initializing Vectors")
jsonVectorObject = [{},{}]
titles = ["train","t10k"]
for j in range(2):
    if not os.path.isfile(f"./VectorData{j}.json"):
        reader = MNIST_READER("./data/",titles[j])
        print("Missing Vector Data Creating File...")
        for i in tqdm (range(0,reader.samples), desc="Calculating Vectors.."):
            label,img = reader.readImage(i)
            jsonVectorObject[j][str(i)] = MNIST_VECTOR_DECIDE.craftVectorObject(label,img)

        with open(f"./VectorData{j}.json", "w") as file:
            file.write(json.dumps(jsonVectorObject[j], indent=2))

        del reader
    else:
        file = open(f"./VectorData{j}.json", "r")
        jsonVectorObject[j] = json.load(file)
        file.close()

print("Vector Data Loaded/Created")

print("Select Mode:")
print("1) Random Sample With Training Data")
print("2) Test With Custom Image Input")
answer = input("(1,2):")
if answer == "1":
    size = input("Enter Random Sample Size(100)")
    if size == "":
        size = "100"
    knn = input("Enter Nearest Neighbors Value(10):")
    if knn == "":
        knn = "10"
    
    sampleResult,sampleSucess,sampleTests = sampleTest(jsonVectorObject[0],jsonVectorObject[1], int(size), int(knn))
    print("Random Sample Result", str(sampleResult)+"%")
    print(f"{sampleSucess} out of {sampleTests} suceeded")
elif answer == "2":
    while True:
        customImagePath = input("Enter Path to Image(Must be 28x28)(test.bmp):")
        if customImagePath == "":
            customImagePath = "test.bmp"
        with Image.open(customImagePath) as im:
            data = cv2.imread(customImagePath,cv2.IMREAD_GRAYSCALE)
            data = cv2.bitwise_not(data)
            output = np.zeros((28,28), dtype=np.float32)
            for idX,x in enumerate(data):
                for idY,y in enumerate(x):
                    output[idX][idY] = y

            prediction = predict(jsonVectorObject[0], img=output,k=10)
            freqTable, total = createFreqTable(prediction)
            print("Was your number?\n"+str(mostFrequent(freqTable)))