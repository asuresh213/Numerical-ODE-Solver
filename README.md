# Numerical ODE-Solver
Python library for solving numerical ODEs, predominantly of the form
                         y = f'(t,y)
using methods such as
1. Runge-Kutta 4th order
2. Runge-Kutta-Fehlberg method
3. Adam Bashforth multi-step explicit methods
4. Adam Moulton multi-step implicit methods
5. Predictor corrector methods (with Adam bashforth as predictor
                                and Adam Moulon as corrector)
6. Adam's variable step size predictor corrector
                              (with  4-step Adam Bashforth as predictor
                                and 3-step Adam Moulon as corrector)



However, the library also supports higher order ODEs, or ODE systems using

1. Runge-Kutta 4th order scheme

to solve the given IVP.

Progress is to be made on implementing all first order other methods to ODE systems


In addition, the library also hosts helper functions to create
1. Plots of solutions
2. Table of solution values; latex formatted
3. Analytic solution calculator at sample values, given analytic function.
