from tkinter import *
from main import *

form = Tk()
form.geometry("800x600")

var = StringVar()
label1 = Label( form, textvariable=var )
var.set("Select Two Classes")
label1.pack(anchor = W )


classesNumbers = IntVar()
R1 = Radiobutton(form, text="BOMBAY & CALI", variable=classesNumbers, value=1)
R1.pack( anchor = W )

R2 = Radiobutton(form, text="BOMBAY & SIRA", variable=classesNumbers, value=2)
R2.pack( anchor = W )

R3 = Radiobutton(form, text="CALI & SIRA", variable=classesNumbers, value=3)
R3.pack( anchor = W)

var = StringVar()
label1 = Label( form, textvariable=var )
var.set("Select Two Features")
label1.pack(anchor = W )

def limit_choices():
    selected_count = sum(var.get() > 0 for var in features)
    if selected_count > 2:
        for var in features:
            if var.get() == 0:
                continue
            elif selected_count > 2:
                var.set(0)
                selected_count -= 1

features = [IntVar() for _ in range(5)]

lst=['Area','Perimeter','MajorAxisLength','MinorAxisLength','roundnes']
F1 = Checkbutton(form, text=lst[0], variable=features[0], command=limit_choices, onvalue=1, offvalue=0)
F2 = Checkbutton(form, text=lst[1], variable=features[1], command=limit_choices, onvalue=2, offvalue=0)
F3 = Checkbutton(form, text=lst[2], variable=features[2], command=limit_choices, onvalue=3, offvalue=0)
F4 = Checkbutton(form, text=lst[3], variable=features[3], command=limit_choices, onvalue=4, offvalue=0)
F5 = Checkbutton(form, text=lst[4], variable=features[4], command=limit_choices, onvalue=5, offvalue=0)


F1.pack(anchor=W)
F2.pack(anchor=W)
F3.pack(anchor=W)
F4.pack(anchor=W)
F5.pack(anchor=W)

var = StringVar()
label1 = Label( form, textvariable=var )
var.set("Enter Learning rate")
label1.pack(anchor = W )

learningRate = DoubleVar()
E1 = Entry(form, bd =5,textvariable=learningRate)
E1.place(x=150,y=242)


var = StringVar()
label1 = Label( form, textvariable=var )
var.set("Enter number of epochs")
label1.place(x=0,y=275 )

epoch = IntVar()
E1 = Entry(form, bd =5,textvariable=epoch)
E1.place(x=150,y=275)

var = StringVar()
label1 = Label( form, textvariable=var )
var.set(" Enter MSE threshold")
label1.place(x=0,y=308 )

mse_th = DoubleVar()
E1 = Entry(form, bd =5,textvariable=mse_th)
E1.place(x=150,y=308)



isBias=IntVar()
C6 = Checkbutton(form, text = "bias", variable = isBias,onvalue = 1, offvalue = 0)
C6.place(x=0,y=340)



var = StringVar()
label1 = Label( form, textvariable=var )
var.set("Choose the used algorithm")
label1.place(x=0,y=370 )

Algorithm = IntVar()
A1 = Radiobutton(form, text="perceptron ", variable=Algorithm, value=1)
A1.place( x=0,y=400 )

A2 = Radiobutton(form, text="Adaline ", variable=Algorithm, value=2)
A2.place(x=0,y=430)

def parameters():
    if classesNumbers.get()==1:
            c1=data[data['Class']=="BOMBAY"].copy()
            c2=data[data['Class']=="CALI"].copy()

    elif classesNumbers.get()==2:
            c1=data[data['Class']=="BOMBAY"].copy()
            c2 = data[data['Class'] == "SIRA"].copy()

    else:
        c1=data[data['Class']=="CALI"].copy()
        c2 = data[data['Class'] == "SIRA"].copy()
    selectFeaures = []
    for i in range(5):
        if features[i].get() != 0:
            selectFeaures.append(i)

    f1 = lst[selectFeaures[0]]
    f2 = lst[selectFeaures[1]]
    lrate=learningRate.get()
    epochs=epoch.get()
    mse_threshold=mse_th.get()
    bias = isBias.get()
    algo = ""
    if Algorithm.get() == 1:
        algo = "perceptron"
    else:
        algo = "Adaline"

    return c1,c2,f1,f2,lrate,epochs,mse_threshold,bias,algo

def call_predict():
    c1, c2, f1, f2, lr, epochs, mse, bias, algo=parameters()
    predict(c1, c2, f1, f2, lr, epochs, mse, bias, algo)

B1 = Button(form, text="predict", command=lambda: call_predict())
B1.place(x=400,y=500)



form.mainloop()