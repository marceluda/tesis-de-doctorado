#!/home/lolo/anaconda3/bin/python3
# -*- coding: utf-8 -*-


"""
La única fnción de este archivo es dejar registro de las cuentas 
realizadas para pasar del hamiltoniano de 3 niveles de un átomo 
a la descripción en el sistema rotante.

Tambien se muestra la conversión a la base propia del estado oscuro de CPT
"""

from IPython.display import display



from numpy import *
import matplotlib.pyplot as plt


from time import time




# Cargamos algunas funciones auxiliares


"""
Espacio vectorial en sympy
"""


from sympy import MatrixSymbol, Matrix, symbols, linear_eq_to_matrix, simplify, Rational
from sympy.parsing.sympy_parser import parse_expr
import sympy as sy



N = 3  # Dimensión del problema


# Funcion auziliar para armar matrices 
def s(m,n):
    """
    Representa la matriz/operador:
        |m><n|
    """
    if min(m,n)<0 or max(m,n)>N:
        raise ValueError(f'n,m deben estar entre 0 y {N}')
    rta = zeros((N,N))
    rta[m,n] = 1 
    return Matrix(rta)



#%% Base canónica y trasformación a sistema rotante


Omega_0,Omega_1   = symbols('Omega_0,Omega_1', positive=True)
omega_0,omega_1   = symbols('omega_0,omega_1', positive=True)

omega_01,omega_02 = symbols('omega_01,omega_02', positive=True)
t                 = symbols('t', positive=True)


H0  = Matrix( zeros((N,N)))
H1  = Matrix( zeros((N,N)))


H0 +=  omega_01                    *s(1,1) + omega_02                    *s(2,2)
H1 +=  Omega_0/2*sy.cos(omega_0*t) *s(0,2) + Omega_1/2*sy.cos(omega_1*t) *s(1,2)
H1 += H1.transpose()


print('Estos son los términos del hamiltoniano canónico:')

display(sy.Eq(symbols('H_0'),H0, evaluate=False))
display(sy.Eq(symbols('H_1'),H1, evaluate=False))

print('Matrices para la transformación a sistema rotante:')

B   = Matrix( zeros((N,N)))
B  += (omega_01-omega_0+omega_1)/2  * s(0,0)
B  += (omega_01+omega_0-omega_1)/2  * s(1,1)
B  += (omega_01+omega_0+omega_1)/2  * s(2,2)

U   = sy.exp(-1j*B*t)

display(sy.Eq(symbols('B'),B, evaluate=False))
display(sy.Eq(symbols('U'),U, evaluate=False))


print('En el sistema rotante queda:')

display(sy.Eq(symbols('H_0'), simplify( H0-B          )   , evaluate=False))
display(sy.Eq(symbols('H_1'), simplify( U.inv()*H1*U  )   , evaluate=False))


print('Podemos reescribir el hamiltoniano en términos de :')


delta_0,delta_1   = symbols('delta_0,delta_1', real=True)
delta  ,Delta     = symbols('delta,Delta', real=True)

sol = sy.solve( [ omega_0-omega_02-delta_0,  omega_1-(omega_02-omega_01)-delta_1 ],
                  omega_01 , omega_02      )

for kk,vv in sol.items():
    display(sy.Eq( kk , vv  , evaluate=False))

print('Y queda:')

display(sy.Eq(symbols('H_0'), (H0-B).subs(sol)   , evaluate=False))



print('O en función de:')

sol2= sy.solve( [ delta_0-delta_1-Delta,  (delta_0+delta_1)/2-delta ],
                  delta_0 , delta_1      )

for kk,vv in sol2.items():
    display(sy.Eq( kk , vv  , evaluate=False))

print('Y queda:')

H0 = (H0-B).subs(sol).subs(sol2)
display(sy.Eq(symbols('H_0'),  H0   , evaluate=False))



print('Luego se realiza la Rotating wave approximation para H1:')

H1  =  Omega_0/2 *s(0,2) + Omega_1/2 *s(1,2)
H1 +=  H1.transpose()

display(sy.Eq(symbols('H_1'), H1   , evaluate=False))




#%% Cambio de base a estados Dark y Bright

print('Realizo un cambio de base con el operador:')

S = Omega_1/sy.sqrt(Omega_0**2+Omega_1**2)
C = Omega_0/sy.sqrt(Omega_0**2+Omega_1**2)

U = s(0,1)*S+s(0,0)*C +s(1,1)*C - s(1,0)*S + 1*s(2,2)


display(sy.Eq(symbols('U'), U   , evaluate=False))

print('Lo que me lelva a nuevas expresiones para H0 y H1 en la nueva base:')

display(sy.Eq(symbols('H_0'), simplify(U*H0*U.inv())   , evaluate=False))

display(sy.Eq(symbols('H_1'), simplify(U*H1*U.inv())   , evaluate=False))
















