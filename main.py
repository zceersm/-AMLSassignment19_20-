from A1.taskA1 import *
from A2.taskA2 import *
from B1.taskB1 import *
from B2.taskB2 import *

# ==================================================================================================================
# Data preprocessing
data_trainA2, data_valA2, data_testA2 = data_preprocessingA2()
data_trainB2, data_valB2, data_testB2 = data_preprocessingB2()
# ==================================================================================================================
# Task A1
model_A1 = buildA1()
acc_A1_train = trainA1(model_A1)
acc_A1_test = testA1(model_A1)
del model_A1
# ==================================================================================================================
# Task A2
model_A2 = buildA2()
acc_A2_train = trainA2(model_A2, data_trainA2, data_valA2)
acc_A2_test = testA2(model_A2, data_testA2)
del model_A2, data_trainA2, data_valA2, data_testA2
# ==================================================================================================================
# Task B1
model_B1 = buildB1()
acc_B1_train = trainB1(model_B1)
acc_B1_test = testB1(model_B1)
del model_B1
# ==================================================================================================================
# Task B2
model_B2 = buildB2()
acc_B2_train = trainB2(model_B2, data_trainB2, data_valB2)
acc_B2_test = testB2(model_B2, data_testB2)
del model_B2, data_trainB2, data_valB2, data_testB2
# ==================================================================================================================
# Print results
print('TA1:{}, {}; TA2:{}, {}; TB1:{}, {}; TB2:{}, {};'.format(acc_A1_train, acc_A1_test,
                                                        acc_A2_train, acc_A2_test,
                                                        acc_B1_train, acc_B1_test,
                                                        acc_B2_train, acc_B2_test))