# B2_Ch8_1.py

###############
# Prepared by Ran An, Wei Lu, and Feng Zhang
# Editor-in-chief: Weisheng Jiang, and Sheng Tu
# Book 2  |  Financial Risk Management with Python
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2021
###############

from prettytable import PrettyTable
   
# position R1
x = PrettyTable(["Payout", "Probability"])
x.add_row([-50, 0.02])
x.add_row([0, 0.98])
x.add_row(['97% VaR', 0])
print(x.get_string(title="Position R1"))

# position R2
x = PrettyTable(["Payout", "Probability"])
x.add_row([-50, 0.02])
x.add_row([0, 0.98])
x.add_row(['97% VaR', 0])
print(x.get_string(title="Position R2"))