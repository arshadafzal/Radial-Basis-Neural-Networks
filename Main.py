#  Demonstration: Run the code
#  Author: Arshad Afzal, IIT Kanpur, India
#  For Questions/ Comments, please email to arshad.afzal@gmail.com
import xlrd
from rbnn import *
from sim import *
#  Export Training Data from Excel Workbook
wb = xlrd.open_workbook('TrainData.xlsx')
sheet = wb.sheet_by_index(0)
q = sheet.nrows
r = sheet.ncols
x_train = np.zeros([sheet.nrows, sheet.ncols-1])
y_train = np.zeros([sheet.nrows, 1])
theta = 10
funtol = 0.01
for i in range(sheet.nrows):
    for j in range(sheet.ncols - 1):
        x_train[i][j] = sheet.cell_value(i, j)
    y_train[i] = sheet.cell_value(i, sheet.ncols - 1)
#  Export Test Data from Excel Workbook
wb = xlrd.open_workbook('TestData.xlsx')
sheet = wb.sheet_by_index(0)
x_test = np.zeros([sheet.nrows, sheet.ncols-1])
for i in range(sheet.nrows):
    for j in range(sheet.ncols - 1):
        x_test[i][j] = sheet.cell_value(i, j)
f = open("Resultfile.txt", "a")
f.write("Hyperparameter theta:\n")
f.write(str(theta))
f.write("\n\n")
f.write("Mean-squared error goal:\n")
f.write(str(funtol))
f.write("\n\n")
#  Train the network
[c, lw] = rbnn(x_train, y_train, theta, funtol)
f.write("Neurons centers:\n")
np.savetxt(f, c)
f.write("\n")
f.write("Linear Weights:\n")
np.savetxt(f, lw)
f.write("\n")
#  Simulate network for prediction
yp = sim(x_train, theta, c, lw)
f.write("Predicted response, y:\n")
np.savetxt(f, yp)
f.close()
