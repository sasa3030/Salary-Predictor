"""********************************GUI CODE**************************"""
import numpy as np
import pandas as pd
from tkinter import *

root=Tk()

z0=["GOVERNMENT","BACHELORS","ADMINISTRATION","DIVORCED","FEMALE"]
z1=["PRIVATE","DOCTORATE","MARRIED",'BACKEND DEVELOPER','MALE']
z2=["SELF EMPLOYED","HIGH SCHOOL","CLERK","UNMARRIED"]
z3=["MASTERS","WIDOWED","CRAFTSMAN"]
z4=["OTHERS","EXECUTIVE MANAGER"]
z5=["SENIOR SECONDARY",'FRONTEND DEVELOPER']
z6=["FULL STACK DEVELOPER"]
def check():
    data=[]
    data.append(e1.get())
    data.append(e2.get())
    data.append(e3.get())
    data.append(e4.get())
    data.append(e5.get())
    data.append(e6.get())
    data.append(e7.get())
    data.append(e8.get())
    for i in range(len(data)):
        if(i==0 or i==3 or i==7):
            data[i]=int(data[i])
        elif(i==1 or i==2 or i==4 or i==5 or i==6):
            if(data[i].upper() in z0 ):
                data[i]=0
            elif(data[i].upper() in z1 ):
                data[i]=1
            elif(data[i].upper() in z2 ):
                data[i]=2
            elif(data[i].upper() in z3 ):
                data[i]=3
            elif(data[i].upper() in z4 ):
                data[i]=4
            elif(data[i].upper() in z5 ):
                data[i]=5
            elif(data[i].upper() in z6 ):
                data[i]=6
            else:
                messagebox.showerror("Error", "INVALID ENTRY")
        else:
            pass


    data=np.array(data).reshape(1,-1)
    y=lm.predict(data)
    print(y)
    l9=Label(root,text="Your salary will be : INR"+str(y[0]),
             width=40,font=60,bg="green")
    l9.grid()


l1=Label(root,text="Age",width=20,font=40)
l1.grid()
e1=Entry(root,bg="chartreuse2",font=40)
e1.grid()
l2=Label(root,text="Work Class",width=20,font=40)
l2.grid()
e2=Entry(root,bg="chartreuse2",font=40)
e2.grid()
l3=Label(root,text="Education",width=20,font=40)
l3.grid()
e3=Entry(root,bg="chartreuse2",font=40)
e3.grid()
l4=Label(root,text="Experience in years",width=20,font=40)
l4.grid()
e4=Entry(root,bg="chartreuse2",font=40)
e4.grid()
l5=Label(root,text="Martial status",width=20,font=40)
l5.grid()
e5=Entry(root,bg="chartreuse2",font=40)
e5.grid()
l6=Label(root,text="Occupation",width=20,font=40)
l6.grid()
e6=Entry(root,bg="chartreuse2",font=40)
e6.grid()
l7=Label(root,text="Gender",width=20,font=40)
l7.grid()
e7=Entry(root,bg="chartreuse2",font=40)
e7.grid()
l8=Label(root,text="Hours per week",width=20,font=40)
l8.grid()
e8=Entry(root,bg="chartreuse2",font=40)
e8.grid()

b1=Button(root,text="Predict",bg="brown",fg="white",
          width=10,font=("bold",10),command=check)
b1.grid()




"""********************************ML CODE**************************"""
import numpy as np
import pandas as pd
df=pd.read_excel(r'C:\Users\dell\Desktop\ML Data\salary.xlsx')
from sklearn.preprocessing import LabelEncoder
Label_encoder = LabelEncoder()
df['Workclass'] = Label_encoder.fit_transform(df['Workclass'])
df['Education'] = Label_encoder.fit_transform(df['Education'])
df['Marital Status'] = Label_encoder.fit_transform(df['Marital Status'])
df['Occupation'] = Label_encoder.fit_transform(df['Occupation'])
df['Sex'] = Label_encoder.fit_transform(df['Sex'])
x=df.drop(['Salary'],axis=1)
print(x.head(3))
y=df['Salary']
print(y.head(3))
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1,random_state=37)
print(x_train.shape,x_test.shape)
print(y_train.shape,y_test.shape)
from sklearn.linear_model import LinearRegression
lm=LinearRegression()
model=lm.fit(x_train,y_train)
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
poly = PolynomialFeatures(degree = 2)
x_poly = poly.fit_transform(x_train) #it will put the values according to polynomial values
poly.fit(x_poly,y_train)
lin2=LinearRegression()
lin2.fit(x_poly, y_train)
#lin2.predict(poly.fit_transform(x_train))
x=[23,1,1,1,5,5,1,28]
x=np.array(x).reshape(1,-1)
"""***************************************************************"""

root.mainloop()
