# Demonstration: Run the RBF Neural Network Code

import numpy as np
import xlrd
from rbnn import rbnn
from sim import sim

# ---------------------------------------------------------
# User-defined hyperparameters
# ---------------------------------------------------------

# RBF shape parameter
theta = 1

# Mean squared error tolerance
funtol = 0.000001

# ---------------------------------------------------------
# Read training data from Excel workbook
# ---------------------------------------------------------
wb = xlrd.open_workbook('Train_Data.xls')
sheet = wb.sheet_by_index(0)
q = sheet.nrows
r = sheet.ncols - 1
x_train = np.zeros([q, r])
y_train = np.zeros([q, 1])

for i in range(q):
    for j in range(r):
        x_train[i][j] = sheet.cell_value(i, j)
    y_train[i] = sheet.cell_value(i, r)

# ---------------------------------------------------------
# Read test data from Excel workbook
# ---------------------------------------------------------
wb = xlrd.open_workbook('Train_Data.xls')
sheet = wb.sheet_by_index(1)
x_test = np.zeros([sheet.nrows, sheet.ncols])
for i in range(sheet.nrows):
    for j in range(sheet.ncols):
        x_test[i][j] = sheet.cell_value(i, j)
# ---------------------------------------------------------
# Open result file
# ---------------------------------------------------------
f = open("Resultfile.txt", "w")
# ---------------------------------------------------------
# Write hyperparameter information
# ---------------------------------------------------------
f.write("Hyperparameter theta:\n")
f.write(str(theta))
f.write("\n\n")
f.write("Mean-squared error goal:\n")
f.write(str(funtol))
f.write("\n\n")
# ---------------------------------------------------------
# Train the RBF neural network
# ---------------------------------------------------------
[c, lw] = rbnn(x_train, y_train, theta, funtol)
f.write("Neurons centers:\n")
np.savetxt(f, c)
f.write("\n")
f.write("Linear Weights:\n")
np.savetxt(f, lw)
f.write("\n")
# ---------------------------------------------------------
# Simulate network for prediction
# ---------------------------------------------------------
yp = sim(x_test, theta, c, lw)
print(yp)
f.write("Predicted response, y:\n")
np.savetxt(f, yp)
f.close()

print("Training completed successfully.")
print("Results saved in Resultfile.txt")
