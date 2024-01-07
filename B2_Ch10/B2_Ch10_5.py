# B2_Ch10_5.py

###############
# Prepared by Ran An, Wei Lu, and Feng Zhang
# Editor-in-chief: Weisheng Jiang, and Sheng Tu
# Book 2  |  Financial Risk Management with Python
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2021
###############

B2_Ch10_5_A.py 
# import libraries
import numpy as np
import matplotlib.pyplot as plt
import QuantLib as ql

# set evaluation date
today = ql.Date(15,10,2020)
ql.Settings.instance().setEvaluationDate = today
ql.Settings.instance().evaluationDate = today

# set Marketdata
rate = ql.SimpleQuote(0.025)
rate_handle = ql.QuoteHandle(rate)
dc = ql.Actual365Fixed()
crv = ql.FlatForward(today, rate_handle, dc)
crv.enableExtrapolation()
yts = ql.YieldTermStructureHandle(crv)
hyts = ql.RelinkableYieldTermStructureHandle(crv)
index = ql.Euribor6M(hyts)



B2_Ch10_5_B.py
# set a swap
start = today + ql.Period("2d")
maturity = ql.Period("4Y")
end = ql.TARGET().advance(start, maturity)
nominal = 3e8
fixedRate = 0.025
typ = ql.VanillaSwap.Receiver
spread = 0.0

start1 = today - ql.Period("2d")
maturity1 = ql.Period("5Y")
end1 = ql.TARGET().advance(start, maturity1)

fixedSchedule = ql.Schedule(start,
                            end, 
                            ql.Period("1y"), 
                            index.fixingCalendar(), 
                            ql.ModifiedFollowing,
                            ql.ModifiedFollowing, 
                            ql.DateGeneration.Backward,
                            False)
floatSchedule = ql.Schedule(start,
                            end,
                            index.tenor(),
                            index.fixingCalendar(),
                            index.businessDayConvention(),
                            index.businessDayConvention(),
                            ql.DateGeneration.Backward,
                            False)
floatSchedule1 = ql.Schedule(start1,
                            end1,
                            index.tenor(),
                            index.fixingCalendar(),
                            index.businessDayConvention(),
                            index.businessDayConvention(),
                            ql.DateGeneration.Backward,
                            False)
swap = ql.VanillaSwap(typ, 
                      nominal,
                      fixedSchedule,
                      fixedRate,
                      ql.Thirty360(ql.Thirty360.BondBasis),
                      floatSchedule,
                      index,
                      spread,
                      index.dayCounter())

# pricing engine and npv
engine = ql.DiscountingSwapEngine(hyts)
swap.setPricingEngine(engine)
swap.NPV()
print(swap.NPV())



B2_Ch10_5_C.py 
# model parameters
vol = [ql.QuoteHandle(ql.SimpleQuote(0.008)),
         ql.QuoteHandle(ql.SimpleQuote(0.008))]
meanRev = [ql.QuoteHandle(ql.SimpleQuote(0.04)),
           ql.QuoteHandle(ql.SimpleQuote(0.04))]
model = ql.Gsr(yts, [today+365], vol, meanRev) 
process = model.stateProcess()



B2_Ch10_5_D.py 
# evaluation time grid
date_grid = [today + ql.Period(i,ql.Months) for i in range(0,12*5)]
fixingDate = [index.fixingDate(x) for x in floatSchedule][:-1]
date_grid += fixingDate
date_grid = np.unique(np.sort(date_grid))
time_grid = np.vectorize(lambda x: ql.ActualActual(ql.ActualActual.Bond, floatSchedule1).yearFraction(today, x))(date_grid)
dt = time_grid[1:] - time_grid[:-1]



B2_Ch10_5_E.py 
# random number generator
seed = 666
urng = ql.MersenneTwisterUniformRng(seed)
usrg = ql.MersenneTwisterUniformRsg(len(time_grid)-1,urng)
rn_generator = ql.InvCumulativeMersenneTwisterGaussianRsg(usrg)

# MC simulations
sim_num = 1000
x = np.zeros((sim_num, len(time_grid)))
y = np.zeros((sim_num, len(time_grid)))
pillars = np.array([0.0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
zero_bonds = np.zeros((sim_num, len(time_grid), len(pillars)))

for j in range(len(pillars)):
    zero_bonds[:, 0, j] = model.zerobond(pillars[j],0,0)
    
for n in range(0,sim_num):
    dWs = rn_generator.nextSequence().value()
    for i in range(1, len(time_grid)):
        t0 = time_grid[i-1]
        t1 = time_grid[i]
        x[n,i] = process.expectation(t0,x[n,i-1],dt[i-1]) + dWs[i-1] * process.stdDeviation(t0,x[n,i-1],dt[i-1])
        y[n,i] = (x[n,i] - process.expectation(0,0,t1)) / process.stdDeviation(0,0,t1)
        for j in range(len(pillars)):
            zero_bonds[n, i, j] = model.zerobond(t1+pillars[j],t1,y[n, i])

# plot the paths
plt.style.use('ggplot')
for i in range(0,sim_num):
    plt.plot(time_grid, x[i,:])
plt.xlabel("Time in years")
plt.ylabel("Zero rate")
plt.title("Monte Carlo simulation")    



B2_Ch10_5_F.py 
# swap pricing
npv_cube = np.zeros((sim_num,len(date_grid)))
for p in range(0,sim_num):
    for t in range(0, len(date_grid)):
        date = date_grid[t]
        ql.Settings.instance().setEvaluationDate(date)
        ycDates = [date,date + ql.Period(6, ql.Months)] 
        ycDates += [date + ql.Period(i,ql.Years) for i in range(1,11)]
        yc = ql.DiscountCurve(ycDates, 
                              zero_bonds[p, t, :], 
                              ql.Actual365Fixed())
        yc.enableExtrapolation()
        hyts.linkTo(yc)
        if index.isValidFixingDate(date):
            fixing = index.fixing(date)
            index.addFixing(date, fixing)
        npv_cube[p, t] = swap.NPV()
    ql.IndexManager.instance().clearHistories()
ql.Settings.instance().setEvaluationDate(today)
hyts.linkTo(crv)

# alculate credit exposure
exposure = npv_cube.copy()
exposure[exposure<0]=0

# plot first 15 NPV and exposure paths
fig, (ax1, ax2) = plt.subplots(2, 1)
for i in range(0,15):
    ax1.plot(time_grid, npv_cube[i,:])
for i in range(0,15):
    ax2.plot(time_grid, exposure[i,:])
ax1.set_xlabel("Time in years")
ax1.set_ylabel("NPV")
ax1.set_title("(a) First 15 simulated npv paths")
ax2.set_xlabel("Time in years")
ax2.set_ylabel("Exposure")
ax2.set_title("(b) First 15 simulated exposure paths")
plt.tight_layout()



B2_Ch10_5_G.py 
# Calculate expected exposure
ee = np.sum(exposure, axis=0)/sim_num
# Calculate PFE curve (95% quantile)
PFE_curve = np.apply_along_axis(lambda x: np.sort(x)[int(0.95*sim_num)],0,exposure)
# plot expected exposure and PFE
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(time_grid, ee)
ax1.set_xlabel("Time in years")
ax1.set_ylabel("Expected Exposure")
ax1.set_title("(a) Expected Exposure")
ax2.plot(time_grid,PFE_curve)
ax2.set_xlabel("Time in years")
ax2.set_ylabel("PFE")
ax2.set_title("(b) PFE")
plt.tight_layout()



B2_Ch10_5_H.py 
# generate the discount factors
discount_factors = np.vectorize(yts.discount)(time_grid)
# calculate discounted npvs
discounted_cube = np.zeros(npv_cube.shape)
discounted_cube = npv_cube * discount_factors
# calculate discounted exposure
discounted_exposure = discounted_cube.copy()
discounted_exposure[discounted_exposure<0] = 0
# calculate discounted expected exposure
discounted_ee = np.sum(discounted_exposure, axis=0)/sim_num

# plot discounted npv and exposure
fig, (ax1, ax2) = plt.subplots(2, 1)
for i in range(0,15):
    ax1.plot(time_grid, discounted_cube[i,:])
for i in range(0,15):
    ax2.plot(time_grid, discounted_exposure[i,:])
ax1.set_xlabel("Time in years")
ax1.set_ylabel("Discounted npv")
ax1.set_title("(a) First 15 simulated discounted npv paths")
ax2.plot(time_grid,discounted_ee)
ax2.set_xlabel("Time in years")
ax2.set_ylabel("Discounted exposure")
ax2.set_title("(b) First 15 simulated discounted exposure paths")
plt.tight_layout()



B2_Ch10_5_I.py 
# plot discounted expected exposure
fig, ax1 = plt.subplots(1, 1)
ax1.plot(time_grid,discounted_ee)
ax1.set_xlabel("Time in years")
ax1.set_ylabel("Discounted expected exposure")
ax1.set_title("(b) Discounted expected exposure")



B2_Ch10_5_J.py 
# build default curve 
pd_dates =  [today + ql.Period(i, ql.Years) for i in range(11)]
hzrates = [0.03 * i for i in range(11)]
pd_curve = ql.HazardRateCurve(pd_dates,hzrates,ql.Actual365Fixed())
pd_curve.enableExtrapolation()
# calculate default probs on grid and plot curve
times = np.linspace(0,25,100)
dp = np.vectorize(pd_curve.defaultProbability)(times)
sp = np.vectorize(pd_curve.survivalProbability)(times)
dd = np.vectorize(pd_curve.defaultDensity)(times)
hr = np.vectorize(pd_curve.hazardRate)(times)
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
ax1.plot(times, dp)
ax2.plot(times, sp)
ax3.plot(times, dd)
ax4.plot(times, hr)
ax1.set_xlabel("Time in years")
ax2.set_xlabel("Time in years")
ax3.set_xlabel("Time in years")
ax4.set_xlabel("Time in years")
ax1.set_ylabel("Probability")
ax2.set_ylabel("Probability")
ax3.set_ylabel("Density")
ax4.set_ylabel("Hazard rate")
ax1.set_title("Default probability")
ax2.set_title("Survival probability")
ax3.set_title("Default density")
ax4.set_title("Hazard rate")



B2_Ch10_5_K.py 
# calculate default probs
PD_vec = np.vectorize(pd_curve.defaultProbability)
dPD = PD_vec(time_grid[:-1], time_grid[1:])
# calculate CVA
RR = 0.4
CVA = (1-RR) * np.sum(discounted_ee[1:] * dPD)
print ("CVA value: %.2f" % CVA)
