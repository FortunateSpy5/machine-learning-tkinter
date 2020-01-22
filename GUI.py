from tkinter import Frame, LabelFrame, StringVar, IntVar, Label, Tk, Entry, Button, TclError, Scrollbar, Toplevel, Canvas, Checkbutton, Radiobutton
from tkinter.constants import *
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


class MachineLearning:
    def __init__(self):
        self.data = None
        self.table = None
        self.selection_x = None
        self.selection_y = None
        self.X = None
        self.y = None
        self.X_test_l = None
        self.X_train_l = None
        self.y_test_l = None
        self.y_train_l = None
        self.X_test = None
        self.X_train = None
        self.y_test = None
        self.y_train = None
        self.le = LabelEncoder()

        self.linreg_model = None
        self.linreg_predictions = None
        self.logreg_model = None
        self.logreg_predictions = None
        self.dtree_model = None
        self.dtree_predictions = None
        self.rforest_model = None
        self.rforest_predictions = None

        self.window = Tk()
        self.color = 'grey95'
        self.window.geometry('620x700')
        self.window.resizable(False, False)
        self.window.configure(background=self.color)
        self.window.title('Machine Learning')
        self.window.iconbitmap('py.ico')

        self.heading = Label(self.window, text="Machine Learning", bg=self.color, pady=20,
                             font=("Helvetica", 35, "bold"))
        self.heading.place(width=620, height=100, bordermode=OUTSIDE, x=0, y=0)

        # File Selection and viewing
        self.frame = LabelFrame(self.window, text='File Selection', bg=self.color)
        self.frame.place(width=580, height=80, bordermode=OUTSIDE, x=20, y=100)

        self.name_label = Label(self.frame, text="File Name : ", bg=self.color, padx=10, pady=10,
                                font=("Helvetica", 15))
        self.name_label.place(width=120, height=30, bordermode=INSIDE, x=10, y=13)

        self.name = StringVar()
        self.name_entry = Entry(self.frame, exportselection=False, textvariable=self.name, font=("Helvetica", 12))
        self.name_entry.place(width=250, height=30, bordermode=INSIDE, x=130, y=13)

        self.name_select = Button(self.frame, text='Select', command=lambda: self.select())
        self.name_select.place(width=50, height=30, bordermode=INSIDE, x=395, y=13)

        self.df_show = Button(self.frame, text='Show', command=lambda: self.create_table(), state=DISABLED)
        self.df_show.place(width=50, height=30, bordermode=INSIDE, x=455, y=13)

        self.df_hide = Button(self.frame, text='Hide', command=lambda: self.hide(), state=DISABLED)
        self.df_hide.place(width=50, height=30, bordermode=INSIDE, x=515, y=13)

        # Train Test Split
        self.ttsplit = LabelFrame(self.window, text='Train Test Split', bg=self.color)
        self.ttsplit.place(width=580, height=80, bordermode=OUTSIDE, x=20, y=200)

        self.select_x = Button(self.ttsplit, text='X', command=lambda: self.get_x(), state=DISABLED)
        self.select_x.place(width=80, height=30, bordermode=INSIDE, x=10, y=13)

        self.select_y = Button(self.ttsplit, text='y', command=lambda: self.get_y(), state=DISABLED)
        self.select_y.place(width=80, height=30, bordermode=INSIDE, x=100, y=13)

        self.test_size_label = Label(self.ttsplit, text="Test Size : ", bg=self.color)
        self.test_size_label.place(width=60, height=30, bordermode=INSIDE, x=200, y=13)

        self.test_size = StringVar()
        self.test_size.set('0.25')
        self.test_size_entry = Entry(self.ttsplit, exportselection=False, textvariable=self.test_size,
                                     font=("Helvetica", 10))
        self.test_size_entry.place(width=50, height=30, bordermode=INSIDE, x=260, y=13)

        self.rstate_label = Label(self.ttsplit, text="Random State : ", bg=self.color)
        self.rstate_label.place(width=100, height=30, bordermode=INSIDE, x=330, y=13)

        self.rstate = StringVar()
        self.rstate.set('None')
        self.rstate_entry = Entry(self.ttsplit, exportselection=False, textvariable=self.rstate, font=("Helvetica", 10))
        self.rstate_entry.place(width=50, height=30, bordermode=INSIDE, x=430, y=13)

        self.split_button = Button(self.ttsplit, text='Split', command=lambda: self.split(), state=DISABLED)
        self.split_button.place(width=80, height=30, bordermode=INSIDE, x=490, y=13)

        # Linear Regression
        self.linreg = LabelFrame(self.window, text='Linear Regression', bg=self.color)
        self.linreg.place(width=580, height=80, bordermode=OUTSIDE, x=20, y=300)

        self.linreg_pred = Button(self.linreg, text='Predict', command=lambda: self.pred_linreg(), state=DISABLED)
        self.linreg_pred.place(width=125, height=30, bordermode=INSIDE, x=8, y=13)

        self.coefficients = Button(self.linreg, text='Coefficients', command=lambda: self.coeff(), state=DISABLED)
        self.coefficients.place(width=125, height=30, bordermode=INSIDE, x=153, y=13)

        self.scatter_button = Button(self.linreg, text='Scatter Plot', command=lambda: self.scatter(), state=DISABLED)
        self.scatter_button.place(width=125, height=30, bordermode=INSIDE, x=298, y=13)

        self.linreg_error = Button(self.linreg, text='Error', command=lambda: self.errors_linreg(), state=DISABLED)
        self.linreg_error.place(width=125, height=30, bordermode=INSIDE, x=443, y=13)

        # Logistic Regression
        self.logreg = LabelFrame(self.window, text='Logistic Regression', bg=self.color)
        self.logreg.place(width=580, height=80, bordermode=OUTSIDE, x=20, y=400)

        self.logreg_pred = Button(self.logreg, text='Predict', command=lambda: self.pred_logreg(), state=DISABLED)
        self.logreg_pred.place(width=125, height=30, bordermode=INSIDE, x=8, y=13)

        self.logreg_cm = Button(self.logreg, text='Confusion Matrix', command=lambda: self.cm_logreg(), state=DISABLED)
        self.logreg_cm.place(width=125, height=30, bordermode=INSIDE, x=153, y=13)

        self.logreg_cr = Button(self.logreg, text='Classification Report', command=lambda: self.cr_logreg(), state=DISABLED)
        self.logreg_cr.place(width=125, height=30, bordermode=INSIDE, x=298, y=13)

        self.logreg_error = Button(self.logreg, text='Error', command=lambda: self.errors_logreg(), state=DISABLED)
        self.logreg_error.place(width=125, height=30, bordermode=INSIDE, x=443, y=13)

        # Decision Tree
        self.dtree = LabelFrame(self.window, text='Decision Tree', bg=self.color)
        self.dtree.place(width=580, height=80, bordermode=OUTSIDE, x=20, y=500)

        self.dtree_pred = Button(self.dtree, text='Predict', command=lambda: self.pred_dtree(), state=DISABLED)
        self.dtree_pred.place(width=125, height=30, bordermode=INSIDE, x=8, y=13)

        self.dtree_cm = Button(self.dtree, text='Confusion Matrix', command=lambda: self.cm_dtree(), state=DISABLED)
        self.dtree_cm.place(width=125, height=30, bordermode=INSIDE, x=153, y=13)

        self.dtree_cr = Button(self.dtree, text='Classification Report', command=lambda: self.cr_dtree(), state=DISABLED)
        self.dtree_cr.place(width=125, height=30, bordermode=INSIDE, x=298, y=13)

        self.dtree_error = Button(self.dtree, text='Error', command=lambda: self.errors_dtree(), state=DISABLED)
        self.dtree_error.place(width=125, height=30, bordermode=INSIDE, x=443, y=13)

        # Random Forest
        self.rforest = LabelFrame(self.window, text='Random Forest', bg=self.color)
        self.rforest.place(width=580, height=80, bordermode=OUTSIDE, x=20, y=600)

        self.rforest_pred = Button(self.rforest, text='Predict', command=lambda: self.pred_rforest(), state=DISABLED)
        self.rforest_pred.place(width=125, height=30, bordermode=INSIDE, x=8, y=13)

        self.rforest_cm = Button(self.rforest, text='Confusion Matrix', command=lambda: self.cm_rforest(), state=DISABLED)
        self.rforest_cm.place(width=125, height=30, bordermode=INSIDE, x=153, y=13)

        self.rforest_cr = Button(self.rforest, text='Classification Report', command=lambda: self.cr_rforest(), state=DISABLED)
        self.rforest_cr.place(width=125, height=30, bordermode=INSIDE, x=298, y=13)

        self.rforest_error = Button(self.rforest, text='Error', command=lambda: self.errors_rforest(), state=DISABLED)
        self.rforest_error.place(width=125, height=30, bordermode=INSIDE, x=443, y=13)

        self.window.mainloop()

    def select(self):
        try:
            self.data = pd.read_csv(self.name.get())
            self.df_show['state'] = NORMAL
            self.df_hide['state'] = NORMAL
            self.name_entry['state'] = DISABLED
            self.name_select['state'] = DISABLED
            self.select_x['state'] = NORMAL
        except FileNotFoundError:
            self.name.set("Invalid")

    def create_table(self):
        try:
            self.table.window.deiconify()
        except AttributeError:
            if self.data.shape[0] > 50:
                self.table = Table(self.data.head(50), self.window, self.name.get())
            else:
                self.table = Table(self.data, self.window, self.name.get())
        except TclError:
            if self.data.shape[0] > 50:
                self.table = Table(self.data.head(50), self.window, self.name.get())
            else:
                self.table = Table(self.data, self.window, self.name.get())

    def hide(self):
        try:
            self.table.window.withdraw()
        except TclError:
            return
        except AttributeError:
            return

    def get_x(self):
        self.selection_x = SelectionX(self.window, self.data)
        self.X = []
        for i in range(len(self.data.columns)):
            if self.selection_x.variables[i].get() == 1:
                self.X.append(self.data.columns[i])

        self.select_x['state'] = DISABLED
        self.select_y['state'] = NORMAL

    def get_y(self):
        self.selection_y = SelectionY(self.window, self.data)
        self.y = self.data.columns[self.selection_y.variable.get()]
        if self.y not in self.X:
            self.split_button['state'] = NORMAL
            self.select_y['state'] = DISABLED

    def split(self):
        test_size = 0.25
        try:
            test_size = float(self.test_size.get())
            if test_size <= 0 or test_size >= 1:
                test_size = 0.25
        except ValueError:
            test_size = 0.25
            self.test_size.set('0.25')
        random_state = None
        if self.rstate.get() != 'None':
            try:
                random_state = int(self.rstate.get())
            except ValueError:
                random_state = None
                self.rstate.set('None')

        self.X_train_l, self.X_test_l, self.y_train_l, self.y_test_l = train_test_split(self.data[self.X], self.data[self.y], test_size=test_size, random_state=random_state)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data[self.X], self.le.fit_transform(self.data[self.y]), test_size=test_size, random_state=random_state)

        self.linreg_pred['state'] = NORMAL
        self.coefficients['state'] = DISABLED
        self.scatter_button['state'] = DISABLED
        self.linreg_error['state'] = DISABLED

        self.logreg_pred['state'] = NORMAL
        self.logreg_cr['state'] = DISABLED
        self.logreg_cm['state'] = DISABLED
        self.logreg_error['state'] = DISABLED

        self.dtree_pred['state'] = NORMAL
        self.dtree_cr['state'] = DISABLED
        self.dtree_cm['state'] = DISABLED
        self.dtree_error['state'] = DISABLED

        self.rforest_pred['state'] = NORMAL
        self.rforest_cm['state'] = DISABLED
        self.rforest_cr['state'] = DISABLED
        self.rforest_error['state'] = DISABLED

    def pred_linreg(self):
        self.linreg_model = LinearRegression()
        self.linreg_model.fit(self.X_train_l, self.y_train_l)
        self.linreg_predictions = self.linreg_model.predict(self.X_test_l)

        self.linreg_error['state'] = NORMAL
        self.scatter_button['state'] = NORMAL
        self.coefficients['state'] = NORMAL

    def scatter(self):
        Scatter(self.window, self.y_test_l, self.linreg_predictions)

    def coeff(self):
        Coefficients(self.window, self.linreg_model.intercept_, self.linreg_model.coef_, self.X)

    def errors_linreg(self):
        temp = [mean_absolute_error(self.y_test, self.linreg_predictions), mean_squared_error(self.y_test, self.linreg_predictions), np.sqrt(mean_squared_error(self.y_test, self.linreg_predictions))]
        Errors(self.window, temp, 'Linear Regression')

    def pred_logreg(self):
        self.logreg_model = LogisticRegression(solver='liblinear')
        self.logreg_model.fit(self.X_train, self.y_train)
        self.logreg_predictions = self.logreg_model.predict(self.X_test)

        self.logreg_cr['state'] = NORMAL
        self.logreg_cm['state'] = NORMAL
        self.logreg_error['state'] = NORMAL

    def cm_logreg(self):
        ConfusionMatrix(self.window, confusion_matrix(self.le.inverse_transform(self.y_test), self.le.inverse_transform(self.logreg_predictions)), 'Logistic Regression', self.le.classes_)

    def cr_logreg(self):
        ClassificationReport(self.window, classification_report(self.le.inverse_transform(self.y_test), self.le.inverse_transform(self.logreg_predictions)), 'Logistic Regression')

    def errors_logreg(self):
        temp = [mean_absolute_error(self.y_test, self.logreg_predictions), mean_squared_error(self.y_test, self.logreg_predictions), np.sqrt(mean_squared_error(self.y_test, self.logreg_predictions))]
        Errors(self.window, temp, 'Logistic Regression')

    def pred_dtree(self):
        self.dtree_model = DecisionTreeClassifier()
        self.dtree_model.fit(self.X_train, self.y_train)
        self.dtree_predictions = self.dtree_model.predict(self.X_test)

        self.dtree_cr['state'] = NORMAL
        self.dtree_cm['state'] = NORMAL
        self.dtree_error['state'] = NORMAL

    def cm_dtree(self):
        ConfusionMatrix(self.window, confusion_matrix(self.le.inverse_transform(self.y_test), self.le.inverse_transform(self.dtree_predictions)), 'Decision Tree', self.le.classes_)

    def cr_dtree(self):
        ClassificationReport(self.window, classification_report(self.le.inverse_transform(self.y_test), self.le.inverse_transform(self.dtree_predictions)), 'Decision Tree')

    def errors_dtree(self):
        temp = [mean_absolute_error(self.y_test, self.dtree_predictions), mean_squared_error(self.y_test, self.dtree_predictions), np.sqrt(mean_squared_error(self.y_test, self.dtree_predictions))]
        Errors(self.window, temp, 'Decision Tree')

    def pred_rforest(self):
        self.rforest_model = RandomForestClassifier(n_estimators=100)
        self.rforest_model.fit(self.X_train, self.y_train)
        self.rforest_predictions = self.rforest_model.predict(self.X_test)

        self.rforest_cr['state'] = NORMAL
        self.rforest_cm['state'] = NORMAL
        self.rforest_error['state'] = NORMAL

    def cm_rforest(self):
        ConfusionMatrix(self.window, confusion_matrix(self.le.inverse_transform(self.y_test), self.le.inverse_transform(self.rforest_predictions)), 'Random Forest', self.le.classes_)

    def cr_rforest(self):
        ClassificationReport(self.window, classification_report(self.le.inverse_transform(self.y_test), self.le.inverse_transform(self.rforest_predictions)), 'Random Forest')

    def errors_rforest(self):
        temp = [mean_absolute_error(self.y_test, self.rforest_predictions), mean_squared_error(self.y_test, self.rforest_predictions), np.sqrt(mean_squared_error(self.y_test, self.rforest_predictions))]
        Errors(self.window, temp, 'Random Forest')


class Table:
    def __init__(self, data, master, name):
        self.master = master
        self.window = Toplevel(self.master)
        self.data = data
        self.name = name
        self.window.title(self.name)
        self.window.geometry('600x600')
        self.window.minsize(250, 250)

        self.frame = Frame(self.window)
        self.frame.pack(expand=True, fill=BOTH)

        self.canvas = Canvas(self.frame, background='white')

        self.h_scroll = Scrollbar(self.frame, orient=HORIZONTAL, command=self.canvas.xview)
        self.h_scroll.pack(side=BOTTOM, fill=X)
        self.v_scroll = Scrollbar(self.frame, orient=VERTICAL, command=self.canvas.yview)
        self.v_scroll.pack(side=RIGHT, fill=Y)

        self.canvas['xscrollcommand'] = self.h_scroll.set
        self.canvas['yscrollcommand'] = self.v_scroll.set
        self.canvas.pack(expand=True, fill=BOTH)

        self.label_frame = LabelFrame(self.canvas)
        self.canvas.create_window((0, 0), window=self.label_frame, anchor=N + W)

        self.shape = (data.shape[0], data.shape[1])

        Table.add_label(self, 0, 0, '#', font=('Helvetica', 15, 'bold'))
        for j in range(self.shape[1]):
            Table.add_label(self, 0, j + 1, self.data.columns[j], font=('Helvetica', 12, 'bold'))
        self.height = 20
        for i in range(self.shape[0]):
            Table.add_label(self, i + 1, 0, str(i + 1))
            ar = data.iloc[i].values
            for j in range(len(ar)):
                Table.add_label(self, i + 1, j + 1, ar[j])
        self.window.update()
        self.canvas.configure(scrollregion=self.label_frame.bbox(ALL))

    def add_label(self, i, j, text, font=('Helvetica', 10)):
        if j % 2 == 0:
            color = 'white'
        else:
            color = 'antique white'
        label = Label(self.label_frame, text=text, font=font, bg=color)
        label.grid(row=i, column=j, sticky=E+N+W+S)


class SelectionX:
    def __init__(self, master, data):
        self.master = master
        self.data = data
        self.columns = self.data.columns
        self.variables = [IntVar() for _ in range(len(self.columns))]

        self.window = Toplevel(self.master)
        self.window.grab_set()
        self.window.title('Independent Variables')
        self.window.geometry('400x400')
        self.window.minsize(250, 250)

        self.frame = Frame(self.window)
        self.frame.pack(expand=True, fill=BOTH)

        self.canvas = Canvas(self.frame, background='antique white')

        self.v_scroll = Scrollbar(self.frame, orient=VERTICAL, command=self.canvas.yview)
        self.v_scroll.pack(side=RIGHT, fill=Y)

        self.canvas['yscrollcommand'] = self.v_scroll.set
        self.canvas.pack(expand=True, fill=BOTH)

        self.frame2 = Frame(self.canvas, bg='antique white')
        self.canvas.create_window((0, 0), window=self.frame2, anchor=N + W)

        for i in range(len(self.columns)):
            Checkbutton(self.frame2, variable=self.variables[i], text=self.columns[i], bg='antique white').pack(anchor=N+W)

        self.all = Button(self.canvas, text='Select All', height=2, width=10, command=lambda: self.select_all())
        self.all.pack(anchor=E, padx=20, pady=20)

        self.none = Button(self.canvas, text='Select None', height=2, width=10, command=lambda: self.select_none())
        self.none.pack(anchor=E, padx=20, pady=0)

        self.none = Button(self.canvas, text='Confirm', height=2, width=10, command=lambda: self.confirm())
        self.none.pack(anchor=E, padx=20, pady=20)

        self.window.update()

        self.canvas.configure(scrollregion=self.canvas.bbox(ALL))

        self.window.mainloop()

    def select_all(self):
        for i in self.variables:
            i.set(1)

    def select_none(self):
        for i in self.variables:
            i.set(0)

    def confirm(self):
        self.window.grab_release()
        self.window.quit()
        self.window.destroy()


class SelectionY:
    def __init__(self, master, data):
        self.master = master
        self.data = data
        self.columns = self.data.columns
        self.variable = IntVar()

        self.window = Toplevel(self.master)
        self.window.grab_set()
        self.window.title('Dependent Variables')
        self.window.geometry('400x400')
        self.window.minsize(250, 250)

        self.frame = Frame(self.window)
        self.frame.pack(expand=True, fill=BOTH)

        self.canvas = Canvas(self.frame, background='antique white')

        self.v_scroll = Scrollbar(self.frame, orient=VERTICAL, command=self.canvas.yview)
        self.v_scroll.pack(side=RIGHT, fill=Y)

        self.canvas['yscrollcommand'] = self.v_scroll.set
        self.canvas.pack(expand=True, fill=BOTH)

        self.frame2 = Frame(self.canvas, bg='antique white')
        self.canvas.create_window((0, 0), window=self.frame2, anchor=N + W)

        for i in range(len(self.columns)):
            Radiobutton(self.frame2, variable=self.variable, value=i, text=self.columns[i], bg='antique white').pack(anchor=N+W)

        self.none = Button(self.canvas, text='Confirm', height=2, width=10, command=lambda: self.confirm())
        self.none.pack(anchor=E, padx=20, pady=20)

        self.canvas.configure(scrollregion=self.canvas.bbox(ALL))

        self.window.mainloop()

    def confirm(self):
        self.window.grab_release()
        self.window.quit()
        self.window.destroy()


class ConfusionMatrix:
    def __init__(self, master, data, name, labels):
        self.data = data
        self.master = master
        self.name = name
        self.labels = sorted(labels)

        self.total = np.sum(self.data)

        self.window = Toplevel(self.master)
        self.window.title(self.name + " Confusion Matrix")
        self.window.resizable(False, False)

        self.total_label = Label(self.window, text=f'Total = {self.total}', font=('Helvetica', 15, 'bold'), bg='antique white')
        self.total_label.grid(row=0, column=0, sticky=(N, S, E, W))

        for i in range(len(self.labels)):
            if i % 2 == 0:
                color = 'white'
            else:
                color = 'antique white'
            Label(self.window, text=f'Predicted\n{self.labels[i]}', font=('Helvetica', 15, 'bold'), bg=color).grid(row=0, column=i+1, sticky=(N, S, E, W))

        for i in range(len(self.labels)):
            if i % 2 == 0:
                color = 'white'
            else:
                color = 'antique white'
            Label(self.window, text=f'Actual\n{self.labels[i]}', font=('Helvetica', 15, 'bold'), bg=color).grid(row=i+1, column=0, sticky=(N, S, E, W))
            for j in range(len(self.labels)):
                color = ['grey90', 'grey80', 'grey70']
                Label(self.window, text=str(self.data[i][j]), font=('Helvetica', 15, 'bold'), bg=color[(i + j) % 3]).grid(row=i+1, column=j+1, sticky=(N, S, E, W))


class Errors:
    def __init__(self, master, data, name):
        self.master = master
        self.data = data
        self.name = name

        self.window = Toplevel(self.master)
        self.window.title(self.name + " Errors")
        self.window.geometry('500x180')
        self.window.resizable(False, False)

        self.frame = Frame(self.window)
        self.frame.place(width=504, height=184, bordermode=OUTSIDE, x=0, y=0)

        self.text1 = Label(self.frame, text='Mean Absolute Error :', font=('Helvetica', 15, 'bold'), bg='antique white')
        self.text1.place(width=260, height=60, bordermode=INSIDE, x=0, y=0)
        self.text2 = Label(self.frame, text='Mean Squared Error :', font=('Helvetica', 15, 'bold'), bg='white')
        self.text2.place(width=260, height=60, bordermode=INSIDE, x=0, y=60)
        self.text3 = Label(self.frame, text='Root Mean Squared Error: ', font=('Helvetica', 15, 'bold'), bg='antique white')
        self.text3.place(width=260, height=60, bordermode=INSIDE, x=0, y=120)

        self.value1 = Label(self.frame, text=str(data[0]), font=('Helvetica', 15, 'bold'), bg='antique white')
        self.value1.place(width=240, height=60, bordermode=INSIDE, x=260, y=0)
        self.value2 = Label(self.frame, text=str(data[1]), font=('Helvetica', 15, 'bold'), bg='white')
        self.value2.place(width=240, height=60, bordermode=INSIDE, x=260, y=60)
        self.value3 = Label(self.frame, text=str(data[2]), font=('Helvetica', 15, 'bold'), bg='antique white')
        self.value3.place(width=240, height=60, bordermode=INSIDE, x=260, y=120)


class ClassificationReport:
    def __init__(self, master, data, name):
        self.master = master
        self.data = data
        self.name = name

        self.window = Toplevel(self.master)
        self.window.title(self.name + " Classification Report")
        self.window.configure(background='white')
        self.window.resizable(False, False)
        y = 0

        Label(self.window, text='precision', font=('Helvetica', 15, 'bold'), anchor=E, bg='antique white').place(width=100, height=50, bordermode=INSIDE, x=150, y=y)
        Label(self.window, text='recall', font=('Helvetica', 15, 'bold'), anchor=E, bg='white').place(width=100, height=50, bordermode=INSIDE, x=250, y=0)
        Label(self.window, text='f1-score', font=('Helvetica', 15, 'bold'), anchor=E, bg='antique white').place(width=100, height=50, bordermode=INSIDE, x=350, y=y)
        Label(self.window, text='support', font=('Helvetica', 15, 'bold'), anchor=E, bg='white').place(width=100, height=50, bordermode=INSIDE, x=450, y=y)
        y = y + 50

        Label(self.window, bg='antique white').place(width=100, height=10, bordermode=INSIDE, x=150, y=y)
        Label(self.window, bg='antique white').place(width=100, height=10, bordermode=INSIDE, x=350, y=y)
        y = y + 10

        self.ar = self.data.split('\n\n')[1:]
        self.part1 = self.ar[0].split('\n')

        for i in self.part1:
            temp = i.split()
            Label(self.window, text=temp[0], font=('Helvetica', 12, 'bold'), anchor=E, bg='white').place(width=150, height=30, bordermode=INSIDE, x=0, y=y)
            Label(self.window, text=temp[1], font=('Helvetica', 12), anchor=E, bg='antique white').place(width=100, height=30, bordermode=INSIDE, x=150, y=y)
            Label(self.window, text=temp[2], font=('Helvetica', 12), anchor=E, bg='white').place(width=100, height=30, bordermode=INSIDE, x=250, y=y)
            Label(self.window, text=temp[3], font=('Helvetica', 12), anchor=E, bg='antique white').place(width=100, height=30, bordermode=INSIDE, x=350, y=y)
            Label(self.window, text=temp[4], font=('Helvetica', 12), anchor=E, bg='white').place(width=100, height=30, bordermode=INSIDE, x=450, y=y)
            y = y + 30

        Label(self.window, bg='antique white').place(width=100, height=20, bordermode=INSIDE, x=150, y=y)
        Label(self.window, bg='antique white').place(width=100, height=20, bordermode=INSIDE, x=350, y=y)
        y = y + 20

        self.part2 = self.ar[1].split('\n')

        for i in self.part2:
            if i == '':
                continue
            temp = i.split()
            Label(self.window, text=temp.pop(), font=('Helvetica', 12), anchor=E, bg='white').place(width=100, height=30, bordermode=INSIDE, x=450, y=y)
            Label(self.window, text=temp.pop(), font=('Helvetica', 12), anchor=E, bg='antique white').place(width=100, height=30, bordermode=INSIDE, x=350, y=y)
            if len(temp) != 1:
                Label(self.window, text=temp.pop(), font=('Helvetica', 12), anchor=E, bg='white').place(width=100, height=30, bordermode=INSIDE, x=250, y=y)
            if len(temp) != 1:
                Label(self.window, text=temp.pop(), font=('Helvetica', 12), anchor=E, bg='antique white').place(width=100, height=30, bordermode=INSIDE, x=150, y=y)
            else:
                Label(self.window, bg='antique white').place(width=100, height=30, bordermode=INSIDE, x=150, y=y)
            Label(self.window, text=" ".join(temp), font=('Helvetica', 12, 'bold'), anchor=E, bg='white').place(width=150, height=30, bordermode=INSIDE, x=0, y=y)
            y = y + 30

        self.window.geometry('550x'+str(y))


class Scatter:
    def __init__(self, master, y_test, pred):
        self.master = master
        self.y_test = y_test
        self.pred = pred

        self.window = Toplevel(self.master)
        self.window.title("Scatter Plot (y_test vs predictions)")
        self.window.configure(background='white')
        self.window.resizable(False, False)

        self.figure = Figure(figsize=(5, 5), dpi=100)
        self.sub = self.figure.add_subplot(111)
        self.sub.scatter(self.y_test, self.pred, edgecolor='black')
        self.sub.plot()

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.window)
        self.canvas.get_tk_widget().pack()
        self.canvas.draw()


class Coefficients:
    def __init__(self, master, intercept, coef, columns):
        self.master = master
        self.intercept = intercept
        self.coef = coef
        self.columns = columns

        self.window = Toplevel(self.master)
        self.window.title("Intercept and Coefficients")
        self.window.configure(background='white')
        self.window.resizable(False, False)

        self.intercept_label = Label(self.window, text='Intercept :', font=('Helvetica', 15, 'bold'), bg='antique white')
        self.intercept_label.grid(row=0, column=0, sticky=(N, S, E, W))
        self.intercept_value = Label(self.window, text=str(self.intercept), font=('Helvetica', 15), bg='white')
        self.intercept_value.grid(row=0, column=1, sticky=(N, S, E, W))

        self.coefs = Label(self.window, text='Coefficients', font=('Helvetica', 15, 'bold'), bg='white')
        self.coefs.grid(row=1, column=0, columnspan=2, sticky=(N, S, E, W))

        for i in range(len(self.coef)):
            Label(self.window, text=self.columns[i], font=('Helvetica', 12), bg='antique white').grid(row=i+2, column=0, sticky=(N, S, E, W))
            Label(self.window, text=str(self.coef[i]), font=('Helvetica', 12), bg='white').grid(row=i+2, column=1, sticky=(N, S, E, W))


if __name__ == '__main__':
    MachineLearning()
