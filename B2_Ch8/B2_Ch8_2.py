# B2_Ch8_2.py

###############
# Prepared by Ran An, Wei Lu, and Feng Zhang
# Editor-in-chief: Weisheng Jiang, and Sheng Tu
# Book 2  |  Financial Risk Management with Python
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2021
###############

from prettytable import PrettyTable
   
# combination of R1 and R2
# probability of 3 possible payouts
p1 = 0.02*0.02
p2 = 2*0.02*0.98
p3 = round(0.98*0.98, 4)

x = PrettyTable(["Payout", "Probability"])
x.add_row([-100, p1])
x.add_row([-50, p2])
x.add_row([0, p3])
# print(x.get_string(title="Combination of positions R1 and R2"))
x.add_row(['97% VaR', 50])
print(x.get_string(title="Combination of positions R1 and R2"))