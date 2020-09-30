import csv
import tensorflow as tf
import numpy as np

def loadData():
    def convert(*data):
        values = []
        for n in data:
            if n[-1] == "Iris-setosa":
                ans = [1,0,0]
            elif n[-1] == "Iris-versicolor":
                ans = [0,1,0]
            elif n[-1] == "Iris-virginica":
                ans = [0,0,1]
            values.append([np.array(n[:-1]),np.array(ans)])

        return values


    filename = "iris.data"
    data = []
    test= []
    count=count_= 0
    with open(filename, "r") as f:
        reader = list(csv.reader(f))[:-1]
        setosa,versicolor,virginica = [reader[x:x+50] for x in range(0,150,50)]

    for d1,d2,d3 in zip(setosa,versicolor,virginica):
        if count <37:
            for n in convert(d1,d2,d3):
                data.append(n)
            count +=1
        else:
            for n in convert(d1,d2,d3):
                test.append(n)

    return (data,test)


def main():

    learn,test = loadData()
    model = tf.keras.models.Sequential([
                                        tf.keras.layers.Dense(10,input_shape=(4,1)),
                                        tf.keras.layers.Dense(3,activation="sigmoid")
    ])
    model.compile(optimizer="adam",loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    data,ans =  learn#np.array(learn[0]),np.array(learn[1])
    model.fit(data,ans,epochs=5)



if __name__ == '__main__':
    main()
