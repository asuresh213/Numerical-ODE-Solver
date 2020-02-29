import numpy as np
import tabulate
import matplotlib.pyplot as plt


'''
The following "library" is for solving ODEs numerically.
This is done as a side project for the class Numerical Analysis II.

The library contains the following methods
--------------------------------------------------------------------------------
Runge-Kutta 4th order:
    Note: This method runs only for IVP of the form y'=f(t,y) and y(0) = alpha

    Function Name: RK4_method

    Inputs: f: defining function ( from y' = f(t,y) )
            a: starting Time
            b: ending Time
            t: Initial time array [a]
            y: Initial solution array [alpha]
            h: stepsize
            N: (optional) to override the number of iterations
                default: N = 0 for normal RK4.
                        reset N to any finite number i to get i iterations

    Outputs: t: Final time array
             y: Final solution array

--------------------------------------------------------------------------------
Runge-Kutta Fehlberg:
    Function Name: RKF_method

    Inputs: f: defining function ( from y' = f(t,y) )
            a: starting Time
            b: ending Time
            alpha: Initial Condition
            tol: Tolerance
            hmax: maximum step size
            hmin: minimum step size

    Outputs: y: solution array
             t: time array
             step: stepsize array


--------------------------------------------------------------------------------
Adam Bashforth m step method:
    Function Name: adam_bashforth

    Inputs: f: defining function ( from y' = f(t,y) )
            a: starting Time
            b: ending Time
            alpha: Initial Condition
            h: stepsize
            type: # of steps (the 'm' in m-step scheme)
                                m = 2,3,4 or 5
            N = (optional) to override the number of iterations
                default: N = 0 for normal Adam Bashforth explicit
                    reset N to any finite number i to get i iterations.

    Outputs: y: solution array


--------------------------------------------------------------------------------
Adam Moulton m step method: (as a corrector)
    Function Name: adam_moulton

    Inputs: f: defining function ( from y' = f(t,y) )
            t: Initial time array [a]
            y: Initial solution array [alpha]
            h: stepsize
            type: # of steps (the 'm' in m-step scheme)
                                m = 2,3 or 4
            wp: predicted value (using some explicit method) at time i+1

    Outputs: y: corrected solution array until time i+1 [w_0, .. w_(i+1)]

--------------------------------------------------------------------------------

Predictor Corrector scheme (Predictor: Adam Bashforth; Corrector: Adam-Moulton)
    Function Name: predict_correct

    Inputs: f: defining function ( from y' = f(t,y) )
            t: Initial time array [a]
            y: Initial solution array [alpha]
            pred: # steps for predictor (the 'm' in Adam Bashforth m-step)
            corr: # steps for corrector (the 'm' in Adam Moulton m-step)
            alpha: Initial Condition
            a: Initial time
            b: Final time
            h: Step size

    Outputs: y: Solution array

--------------------------------------------------------------------------------

Adams variable stepsize predictor corrector

    Note: This method uses 4 step Adam bashforth and 3 step Adam Moulton
    as Predictor and Corrector Respectively.

    Function Name: Adam_PC_variable_step

    Inputs: f: defining function ( from y' = f(t,y) )
            t: Initial time array [a]
            y: Initial solution array [alpha]
            a: Initial time
            b: Final time
            tol: Tolerance value
            hmax: maximum stepsize
            hmin: minimum stepsize

    Outputs: Index: Array of indices (as time array is not of uniform step size)
             t: Time array
             y: Solution array
             h: Step size array

--------------------------------------------------------------------------------
RK4 for systems (of m equations):
    Note: This function was heavily influenced by the style of Dr. Andasari (BU)
    website: http://people.bu.edu/andasari/courses/numericalpython/python.html

    Function Name: RK4_sys

    Inputs: func: The defining functions written in the following style
                func(t, y):
                    dy = [0, 0, ..., 0]
                    dy[0] = u1' = f1(t, y[j]); j in 1,2,..., m
                    dy[1] = u2' = f2(t, y[j]); j in 1,2,..., m
                    .
                    .
                    .
                    dy[m] = um' = fm(t, y[j]); j in 1,2,..., m
                    return dy
            yinit: Array of initial conditions [u1(a), u2(a), ..., um(a)]
            x_range: The array [a, b]
            h: stepsize

    Outputs:
            [tsol, ysol]:
                ysol: An array of arrays of solutions [u1, u2, ..., um]
                tsol: Time axis

--------------------------------------------------------------------------------
Latex table creator
    Function Name: latexit

    Inputs: cols: # of columns of the table
            headers: list of header names
            *args : all columns as lists (in the same order as headers)

    Outputs: Tabulates given values and prints it in latex table format.

--------------------------------------------------------------------------------

ODE solution plotter
    Function Name: plotit

    Inputs: t: the time axis (time array, or x-values array)
            labels: Array of labels of the plots
            *args: all the arrays (to be plotted) in the same order as headers

    Outputs: Plots all the given arrays against the x-values array

--------------------------------------------------------------------------------

Analytic function values - Calculator
    Note: User has to be provided with the actual analytic Solution


    Function Name: analytic_solution

    Inputs: f: The analytic Solution
            t: Time array

    Outputs: soln: The array of analytic solutions

--------------------------------------------------------------------------------



'''





#---------------------- Runge-Kutta-Fehlberg method ------------------

def RKF_method(f, a, b, alpha, tol, hmax, hmin):
    t = a
    w = alpha
    h = hmax
    flag = 1

    time = [a] #time array
    y = [alpha] #solution array
    step = [hmax] #step size array
    while(flag == 1):
        k1 = h*f(t, w)
        k2 = h*f(t + (1/4)*h, w + (1/4)*k1)
        k3 = h*f(t + (3/8)*h, w + (3/32)*k1 + (9/32)*k2)
        k4 = h*f(t + (12/13)*h, w + (1932/2197)*k1 - (7200/2197)*k2 + (7296/2197)*k3)
        k5 = h*f(t + h, w + (439/216)*k1 - (8)*k2 + (3680/513)*k3 - (845/4104)*k4)
        k6 = h*f(t + (1/2)*h, w - (8/27)*k1 + (2)*k2 - (3544/2565)*k3 + (1859/4104)*k4 - (11/40)*k5)
        R = (1/h)*np.abs((1/360)*k1 - (128/4275)*k3 - (2197/75240)*k4 + (1/50)*k5 + (2/55)*k6)
        if(R <= tol):
            t += h
            w += (25/216)*k1 + (1408/2565)*k3 + (2197/4104)*k4 - (1/5)*k5
            time.append(t)
            step.append(h)
            y.append(w)
        delta = 0.84*(tol/R)**(1/4)
        if(delta <= 0.1):
            h = 0.1*h
            step.append(h)
        elif(delta >= 4):
            h = 4*h
            step.append(h)
        else:
            h = delta*h
            step.append(h)
        if(h > hmax):
            h = hmax
            step[-1] = h
        if(t >= b):
            flag = 0
        elif(t + h > b):
            h = b - t
            step[-1] = h
        elif(h<hmin):
            flag = 0
            print("min h exceeded")
            print("completed unsuccessfully")
    return y, time, step


#--------------------- RK 4th order method ---------------------------

def RK4_method(f, a, b, t, y, h, N=0):
    if(N == 0):
        N = int((b-a)/h)

    for i in range(N):
        k1 = h*f(t[-1], y[-1])
        yp2 = y[-1] + k1/2
        k2 = h*f(t[-1]+h/2, yp2)
        yp3 = y[-1] + k2/2
        k3 = h*f(t[-1]+h/2, yp3)
        yp4 = y[-1] + k3
        k4 = h*f(t[-1]+h, yp4)
        val = (1/6)*(k1 + 2*k2 + 2*k3 + k4)
        y.append(y[-1] + val)
        t.append(t[-1] + h)
    return t, y


#----------------- Adam Bashforth explicit methods --------------------

#Adam Bashforth handler

def adam_bashforth(f, a, b, alpha, h, type, N = 0):
    t = [a]
    y = [alpha]
    t, y = RK4_method(f, a, b, t, y, h, type - 1)
    if(N == 0):
        N = int(((b-a)/h)- (type-1))

    if(type == 2):
        y = abf2(f, y, t, h, N)
    if(type == 3):
        y = abf3(f, y, t, h, N)
    if(type == 4):
        y = abf4(f, y, t, h, N)
    if(type == 5):
        y = abf5(f, y, t, h, N)

    return y

#2 step Adam Bashforth
def abf2(f, y, t, h, N):
    for i in range(N):
        y.append(y[-1] + (h/2)*(3*f(t[-1], y[-1]) - f(t[-2], y[-2])))
        t.append(t[-1] + h)
    return y

#3 step Adam Bashforth
def abf3(f, y, t, h, N):
    for i in range(N):
        y.append(y[-1] + (h/12)*(23*f(t[-1], y[-1]) - 16*f(t[-2], y[-2]) \
                                + 5*f(t[-3], y[-3])))
        t.append(t[-1] + h)
    return y

#4 step Adam Bashforth
def abf4(f, y, t, h, N):
    for i in range(N):
        y.append(y[-1] + (h/24)*(55*f(t[-1], y[-1]) - 59*f(t[-2], y[-2]) \
                                    + 37*f(t[-3], y[-3]) - 9*f(t[-4], y[-4])))
        t.append(t[-1] + h)
    return y

#5 step Adam bashforth
def abf5(f, y, t, h, N):
    for i in range(N):
        y.append(y[-1] + (h/720)*(1901*f(t[-1], y[-1]) - 2774*f(t[-2], y[-2]) \
                                + 2616*f(t[-3], y[-3]) - 1274*f(t[-4], y[-4]) \
                                +251*f(t[-5], y[-5])))
        t.append(t[-1] + h)
    return y


#----------------------------------------------------------------------


#------ Adam Moulton Implicit method (for predictor corrector)----------

#Adam Moulton method handler
def adam_moulton(f, t, y, h, type, wp):

    if(type == 2):
        y = abm2(f, y, t, h, 1, wp)
    if(type == 3):
        y = abm3(f, y, t, h, 1, wp)
    if(type == 4):
        y = abm4(f, y, t, h, 1, wp)

    return y

#2 step Adam Moulton
def abm2(f, y, t, h, N, wp):
    for i in range(N):
        y.append(y[-1] + (h/12)*(5*f(t[-1] + h, wp) + 8*f(t[-1], y[-1]) \
                            - f(t[-2], y[-2])))
        t.append(t[-1] + h)
    return y

#3 step Adam Moulton
def abm3(f, y, t, h, N, wp):
    for i in range(N):
        y.append(y[-1] + (h/24)*(9*f(t[-1]+h, wp) + 19*f(t[-1], y[-1]) \
                                - 5*f(t[-2], y[-2]) + f(t[-3], y[-3])))
        t.append(t[-1] + h)
    return y

#4 step Adam Moulton
def abm4(f, y, t, h, N, wp):
    for i in range(N):
        y.append(y[-1] + (h/720)*(251*f(t[-1]+h, wp) + 646*f(t[-1], y[-1]) \
                                - 264*f(t[-2], y[-2]) + 106*f(t[-3], y[-3]) \
                                -19*f(t[-4], y[-4])))
        t.append(t[-1] + h)
    return y


#--------------- Predictor Corrector -------------------------

def predict_correct(f, t, y, pred, corr, alpha, a, b, h):
    N = int(((b-a)/h)- (pred-1))
    t, y = RK4_method(f, a, b, t, y, h, pred - 1)
    for i in range(N):
        #the following predictor is very inefficient.
        #But it works for smaller problems.
        #It can, and will be optimized in further updates

        wp_arr = adam_bashforth(f, a, b, alpha, h, pred, i+1)
        wp = wp_arr[-1]
        y = adam_moulton(f, t, y, h, corr, wp)
    return y

#------------------------------------------------------------
#Adam's Predictor Corrector method with variable step size

def Adam_PC_variable_step(f, t, y, a, b, tol, hmax, hmin):
    def RK4(h, v0, x0):
        x = [x0, 0, 0, 0]
        v = [v0, 0, 0, 0]
        for j in range(1,4):
            k1 = h*f(x[j-1], v[j-1])
            k2 = h*f(x[j-1] + h/2, v[j-1] + k1/2)
            k3 = h*f(x[j-1] + h/2, v[j-1] + k2/2)
            k4 = h*f(x[j-1] + h, v[j-1] + k3)
            v[j] = v[j-1] + (1/6)*(k1 + 2*k2 + 2*k3 + k4)
            x[j] = x[j-1] + h

        return x[1:], v[1:]

    index = [0]
    step = [hmax]
    t0 = a
    w0 = y[0]
    h = hmax
    flag = 1
    last = 0

    tpr, ypr = RK4(h, w0, t0)
    for i in range(len(tpr)):
        t.append(tpr[i])
        y.append(ypr[i])


    nflag = 1
    i = 4
    time = t[i-1] + h

    while(flag == 1):
        wp = y[i-1] + (h/24)*(55*f(t[i-1], y[i-1]) - 59*f(t[i-2], y[i-2]) \
                                    + 37*f(t[i-3], y[i-3]) - 9*f(t[i-4], y[i-4]))
        wc = y[i-1] + (h/24)*(9*f(time, wp) + 19*f(t[i-1], y[i-1]) \
                                - 5*f(t[i-2], y[i-2]) + f(t[i-3], y[i-3]))
        sigma = (19/(270*h))*np.abs(wc - wp)

        if(sigma < tol):

            y.append(wc)
            t.append(time)

            if(nflag == 1):
                for j in range(i-3, i+1):
                    step.append(h)
                    index.append(j)
            else:
                step.append(h)
                index.append(i)

            if(last == 1):
                flag = 0
            else:
                i = i + 1
                nflag = 0

                if(sigma < 0.1*tol or t[i-1] + h > b):

                    q = (tol/(2*sigma))**(1/4)

                    if(q > 4):
                        h = 4*h
                    else:
                        h = q*h

                    if(h > hmax):
                        h = hmax

                    if(t[i-1] + 4*h > b):
                        h = (b - t[i-1])/4
                        last = 1

                    temp1 = []
                    temp2 = []
                    temp1, temp2 = RK4(h, y[i-1], t[i-1])
                    for v in range(len(temp1)):
                        t.append(temp1[v])
                        y.append(temp2[v])
                    nflag = 1
                    i = i+3

        else:

            q = (tol/(2*sigma))**(1/4)


            if(q < 0.1):
                h = 0.1*h
            else:
                h = q*h
            if(h < hmin):
                flag = 0
                print("min h exceeded")
            else:
                if(nflag == 1):
                    i = i-3
                y = y[:i]
                t = t[:i]
                temp3 = []
                temp4 = []
                temp3, temp4 = RK4(h, y[i-1], t[i-1])

                for v in range(len(temp3)):
                    t.append(temp3[v])
                    y.append(temp4[v])
                i = i+3
                nflag = 1
        time = t[i-1] + h
    return index, t, y, step
#------------------------------------------------------------
def RK4_sys(f, yinit, x_range, h):

    m = len(yinit)
    n = int((x_range[-1] - x_range[0])/h)

    x = x_range[0]
    y = yinit

    # Containers for solutions
    xsol = np.empty(0)
    xsol = np.append(xsol, x)

    ysol = np.empty(0)
    ysol = np.append(ysol, y)

    for i in range(n):
        k1 = f(x, y)

        yp2 = y + k1*(h/2)

        k2 = f(x+h/2, yp2)

        yp3 = y + k2*(h/2)

        k3 = f(x+h/2, yp3)

        yp4 = y + k3*h

        k4 = f(x+h, yp4)

        for j in range(m):
            y[j] = y[j] + (h/6)*(k1[j] + 2*k2[j] + 2*k3[j] + k4[j])

        x = x + h
        xsol = np.append(xsol, x)

        for r in range(len(y)):
            ysol = np.append(ysol, y[r])

    solutions = []
    for q in range(m):
        temp = ysol[q::m]
        solutions.append(temp)
        temp = []

    return [xsol, solutions]
#------------------------------------------------------------
#creating latex tables in python!

def latexit(cols, headers, *args):
    N = len(args[0])-1
    table = np.zeros((N+1, cols))
    for i in range(N+1):
        temp = []
        for j in range(len(args)):
            temp.append(args[j][i])
        table[i] = temp
    print(tabulate.tabulate(table, \
        headers = headers,  \
        floatfmt=".8f", tablefmt="latex"))

#--------------------------------------------------------------

def analytic_solution(f, t):
    soln = []
    for time in t:
        soln.append(f(time))
    return soln
#--------------------------------------------------------------

def plotit(name, t, label, *args):
    for i in range(len(args)):
        plt.plot(t, args[i], label = label[i])
    plt.legend()
    plt.show()
    print('''
    \\begin{figure}[h]
	\\centering
	\\includegraphics[scale = 0.75]{%s.png}
	\\caption{Plot of the solution}
    \\end{figure}'''%name)
#--------------------------------------------------------------
