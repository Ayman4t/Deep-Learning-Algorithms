from tkinter import *
from main import *

form = Tk()
form.geometry("600x400")

var = StringVar()
label1 = Label( form, textvariable=var )
var.set("Enter number of hidden layers")
label1.place(x=0,y=10)


hidden = IntVar()
E1 = Entry(form, bd =5,textvariable=hidden)
E1.place(x=170,y=10)


var = StringVar()
label1 = Label( form, textvariable=var )
var.set("Enter number of neurons")
label1.place(x=0,y=50)


neuron = StringVar()
E1 = Entry(form, bd =5,textvariable=neuron)
E1.place(x=170,y=50)


var = StringVar()
label1 = Label( form, textvariable=var )
var.set("Enter learning rate")
label1.place(x=0,y=90)


learnrate = DoubleVar()
E1 = Entry(form, bd =5,textvariable=learnrate)
E1.place(x=170,y=90)


var = StringVar()
label1 = Label( form, textvariable=var )
var.set("Enter number of epochs")
label1.place(x=0,y=130)


epoch = IntVar()
E1 = Entry(form, bd =5,textvariable=epoch)
E1.place(x=170,y=130)


isBias=IntVar()
C6 = Checkbutton(form, text = "bias", variable = isBias,onvalue = 1, offvalue = 0)
C6.place(x=10,y=170)


var = StringVar()
label1 = Label( form, textvariable=var )
var.set("Choose activation function")
label1.place(x=0,y=210 )

activation = IntVar()
A1 = Radiobutton(form, text="Sigmoid", variable=activation, value=1)
A1.place( x=0,y=240 )

A2 = Radiobutton(form, text="Tangent", variable=activation, value=2)
A2.place(x=0,y=270)


def parameters():
    hid=hidden.get()
    neu=neuron.get()
    num_neurons=[]
    s = ""
    for i in neu:
        if i == ',':
            num_neurons.append(int(s))
            s = ""
            continue
        else:
            s += i
    num_neurons.append(int(s))

    lrate = learnrate.get()
    epochs = epoch.get()
    bias = False
    if(isBias.get()==1):
        bias=True
    if activation.get() == 1:
        active = "sigmoid"
    else:
        active = "tangent"

    return hid,num_neurons,lrate,epochs,bias,active


def call_predict():
    hid,num_neurons,lrate,epochs,bias,active=parameters()
    predict(hid,num_neurons,epochs,lrate,bias,active)


B1 = Button(form, text="predict", command= call_predict)
B1.place(x=300,y=300)
form.mainloop()



