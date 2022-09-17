import tkinter as tk
root = tk.Tk()

curr_path = ""
prev_path = ""

def curr():
    global curr_path
    curr_path = entry1.get()
    root.destroy()
def prev():
    global prev_path
    prev_path = entry1.get()
    root.destroy()

#creating first window to enter previous years' dataset
canvas1 = tk.Canvas(root, width = 700, height = 280, bg = "orange1")
canvas1.pack()

label1 = tk.Label(root, text = "  Please enter the link to CSV file containing previous years' employee details  ", bg = "white", font = ("Calibri",14))
canvas1.create_window(350,70, window = label1)

label2 = tk.Label(root, text = "Dataset must contain the employee ID, job role, gender, age, marital status, experience and salary \nof the employee, and must also mention if the employee currently works at the company or not", bg = "orange1", font = ("Calibri",11))
canvas1.create_window(350, 120, window = label2)

entry1 = tk.Entry(root, bg = "white", width = 30, font = "Calibri 15")
canvas1.create_window(350, 170, window = entry1)

tk.Button(root, text = "Next", command = prev, bg = "orange3", font = "Calibri 12").place(x = 330, y = 200)

root.mainloop()

#creating second window to enter current year dataset
root = tk.Tk()

canvas2 = tk.Canvas(root, width = 700, height = 280, bg = "orange1")
canvas2.pack()

label1 = tk.Label(root, text = "  Please enter the link to CSV file containing current year employee details  ", bg = "white", font = ("Calibri",14))
canvas2.create_window(350,70, window = label1)

label2 = tk.Label(root, text = "Dataset must contain the employee ID, job role, gender, age, marital status, experience and salary \nof the employee", bg = "orange1", font = ("Calibri",11))
canvas2.create_window(350, 120, window = label2)

entry1 = tk.Entry(root, bg = "white", width = 30, font = "Calibri 15")
canvas2.create_window(350, 170, window = entry1)

tk.Button(root, text = "Predict", command = curr, bg = "orange3", font = "Calibri 12").place(x = 330, y = 200)

root.mainloop()

#extracting data from the csv file
import pandas as pd
curr_data = pd.read_csv(curr_path)
prev_data = pd.read_csv(prev_path)

#creating ranges for ages,experience and salary
def ran_age(age):
    if age>17 and age<28:
        return "18-27"
    if age>27 and age<41:
        return "28-40"
    if age>40 and age<51:
        return "41-50"
    if age>50 and age<61:
        return "51-60"

def ran_exp(exp):
    if exp>=0 and exp<11:
        return "0-10"
    if exp>=11 and exp<21:
        return "11-20"
    if exp>=21 and exp<31:
        return "21-30"
    if exp>=31 and exp<41:
        return "31-40"

def ran_sal(sal):
    if sal>=2000 and sal<8000:
        return "2k-8k"
    if sal>=8000 and sal<14000:
        return "8k-14k"
    if sal>=14000 and sal<20000:
        return "14k-20k"
    if sal>=20000 and sal<28000:
        return "20k-28k"

#PREVIOUS YEAR DATA - creating columns of ranges for age,experience and salary
prev_data["AgeRange"] = prev_data["Age"].apply(lambda x: ran_age(x))
prev_data["ExperienceRange"] = prev_data["Experience"].apply(lambda x: ran_exp(x))
prev_data["SalaryRange"] = prev_data["MonthlySalary"].apply(lambda x: ran_sal(x))
#CURRENT YEAR DATA - creating columns of ranges for age,experience and salary
curr_data["AgeRange"] = curr_data["Age"].apply(lambda x: ran_age(x))
temp_data = curr_data["AgeRange"]
curr_data["ExperienceRange"] = curr_data["Experience"].apply(lambda x: ran_exp(x))
curr_data["SalaryRange"] = curr_data["MonthlySalary"].apply(lambda x: ran_sal(x))

#dictionaries with numerical values for strings in each column
job_role = {"Sales Executive":1,"Research Scientist":2,"Laboratory Technician":3,"Manufacturing Director":4,"Healthcare Representative":5,"Manager":6,"Sales Representative":7,"Research Director":8,"Human Resources":9}
gender = {"Male":0,"Female":1}
age_range = {"18-27":1,"28-40":2,"41-50":3,"51-60":4}
marital = {"Single":0,"Married":1,"Divorced":2}
exp_range = {"0-10":1,"11-20":2,"21-30":3,"31-40":4}
sal_range = {"2k-8k":1,"8k-14k":2,"14k-20k":3,"20k-28k":4}
attrition = {"Yes":1,"No":0}

#PREVIOUS YEAR DATA - changing string values in columns to numbers using mapping
prev_data["JobRole"] = prev_data["JobRole"].map(job_role)
prev_data["Gender"] = prev_data["Gender"].map(gender)
prev_data["AgeRange"] = prev_data["AgeRange"].map(age_range)
prev_data["MaritalStatus"] = prev_data["MaritalStatus"].map(marital)
prev_data["ExperienceRange"] = prev_data["ExperienceRange"].map(exp_range)
prev_data["SalaryRange"] = prev_data["SalaryRange"].map(sal_range)
prev_data["Attrition"] = prev_data["Attrition"].map(attrition)
#CURRENT YEAR DATA - changing string values in columns to numbers using mapping
curr_data["JobRole"] = curr_data["JobRole"].map(job_role)
curr_data["Gender"] = curr_data["Gender"].map(gender)
curr_data["AgeRange"] = curr_data["AgeRange"].map(age_range)
curr_data["MaritalStatus"] = curr_data["MaritalStatus"].map(marital)
curr_data["ExperienceRange"] = curr_data["ExperienceRange"].map(exp_range)
curr_data["SalaryRange"] = curr_data["SalaryRange"].map(sal_range)

#feature selection
features = ["JobRole","Gender","AgeRange","MaritalStatus","ExperienceRange","SalaryRange"]
x = prev_data[features]
y = prev_data["Attrition"]

#dividing into training and testing data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size = 0.2, random_state = 0)

#training the data using machine learning model
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree = dtree.fit(x_train.values, y_train)

#extracting features and predicting from dataset
attrition = []
for index,row in curr_data.iterrows():
    job = row.JobRole
    gen = row.Gender
    mar = row.MaritalStatus
    age = row.AgeRange
    exp = row.ExperienceRange
    sal = row.SalaryRange
    leaving = int(dtree.predict([[job,gen,mar,age,exp,sal]]))
    attrition.append(leaving)

#creating column for prediction
curr_data = pd.read_csv(curr_path)
curr_data.insert(0,"Attrition",attrition)
att_string = {1:"Yes",0:"No"}
curr_data["Attrition"] = curr_data["Attrition"].map(att_string)

#creating a new dataframe with required columns and converting to csv file format
new_dataframe = curr_data.filter(['EmployeeID','JobRole','Gender','Age','MaritalStatus','Experience','MonthlySalary','Attrition'], axis=1)
new_dataframe.to_csv("C:/Julia/CLEVERED/ADVANCED INTERNSHIP/FINAL PROJECT/Prediction.csv")

#displaying table of prediction
import pandastable as pt

data = pd.read_csv("C:/Julia/CLEVERED/ADVANCED INTERNSHIP/FINAL PROJECT/Prediction.csv")
root = tk.Tk()
root.title('Prediction Table')
canvas =  tk.Canvas(root)
canvas.pack()
pt = pt.Table(canvas, dataframe=data, showtoolbar=False, showstatusbar=False)
pt.show()

#finding number of employees that might leave company
attrition_list = data['Attrition'].tolist()
yes = 0
for i in attrition_list:
    if i == "Yes":
        yes += 1
canvas1 = tk.Canvas(root, width = 200, height = 200)
canvas1.pack()

label1 = tk.Label(root, text = f"Number of employees at risk of attrition: {yes} \n\n Select what plot you would like to view:  ", font = "Calibri 13")
canvas1.create_window(100,50,window = label1)


#creating plots
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def attritionpie():
    plot = data["Attrition"].value_counts().plot(kind = 'pie')
    plt.show()

def jobrolplot():
    plot = data["JobRole"].value_counts().plot(kind = 'bar')
    plt.show()
def genderplot():
    plot = data["Gender"].value_counts().plot(kind = 'bar',color = ['salmon','blue'])
    plt.show()


#creating buttons for plots
tk.Button(root, text = "   Attrition/Retention Pie Chart  ", command = attritionpie, bg = "light blue").place(x = 200, y = 450)
tk.Button(root, text = "   Job Role - No. of Employees   ", command = jobrolplot, bg = "light blue").place(x = 200, y = 480)
tk.Button(root, text = "    Gender - No. of Employees    ", command = genderplot, bg = "light blue").place(x = 200, y = 510)

root.mainloop()

