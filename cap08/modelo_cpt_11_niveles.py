#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Modelo para estudio de CPT con un sistema de 3 niveles

Las ecuaciones están basadas en el trabajo:
    Warren, Z., Shahriar, M. S., Tripathi, R., & Pati, G. S. (2017). 
    Experimental and theoretical comparison of different optical excitation 
    schemes for a compact coherent population trapping Rb vapor clock. Metrologia, 
    54(4), 418–431. 
    https://doi.org/10.1088/1681-7575/aa72bb

"""



from numpy import *
import matplotlib.pyplot as plt

# esto es sólo para reportar los tiemposs de cálculo insumidos
from time import time



#%% Funciones auxiliares y constantes 

a0   = 0.52917720859e-10 # m      Radio de Bohr
hbar = 1.054571628e-34   # J s    h/2 pi
Kmks = 1.3806488e-23     # J/K     Constante de Boltzman
cef  = 1/137.035999046   # Constante de estructura fina
c    =299792458          # m/s  Velocidad de la luz


N = 11  # Dimensión del problema


# Estas son para hacer verificaciones o inspección visual de datos

def ver(M):
    """
    Visualizar Matriz
    """
    fig, ax = plt.subplots(1,1 , figsize=(13,7) )
    margen  = array([0,M.shape[1],M.shape[0],0])+0.5
    im = ax.imshow( abs(M)  ,extent=margen)
    if M.shape[0]<=16:
        ax.set_xticks( arange(1,M.shape[1]+1) )
        ax.set_yticks( arange(1,M.shape[0]+1) )
    ax.set_xlim( 0.5,  M.shape[1]+0.5 )
    ax.set_ylim(  M.shape[0]+0.5 , 0.5 )
    ax.xaxis.set_ticks_position('top')
    #ax.invert_yaxis()
    return ax , im



def Tr(M):
    """
    Calcular Traza
    """
    return sum( diag( M ))




# Funciones auxiliares para ayurdar a construir los 
# operadores de transicion dipolar.
# Reproducen los valores del Steck

def D1_p_fun(F1,m1,F2,m2):
    if not F1 in [1,2]:
        raise ValueError(f'F1 está mal: {F1}')
    if not F2 in [1,2]:
        raise ValueError(f'F2 está mal: {F2}')
    
    if not m2==m1+1:
        return 0
    
    if F1==2:
        if F2==2:
            aa = array([6,4,4,6,0])
            aa = [1/sqrt(a) if abs(a)>0 else 0 for a in aa]
        if F2==1:
            aa = array([2,4,12,0,0])
            aa = [1/sqrt(a) if abs(a)>0 else 0 for a in aa]
        return aa[m1+2]
    
    if F1==1:
        if F2==2:
            aa = array([12,4,2])
            aa = [-1/sqrt(a) if abs(a)>0 else 0 for a in aa]
        if F2==1:
            aa = array([12,12,0])
            aa = [-1/sqrt(a) if abs(a)>0 else 0 for a in aa]
        return aa[m1+1]

def D1_m_fun(F1,m1,F2,m2):
    if not F1 in [1,2]:
        raise ValueError(f'F1 está mal: {F1}')
    if not F2 in [1,2]:
        raise ValueError(f'F2 está mal: {F2}')
    
    if not m2==m1-1:
        return 0
    
    if F1==2:
        if F2==2:
            aa = array([0,6,4,4,6])
            aa = [-1/sqrt(a) if abs(a)>0 else 0 for a in aa]
        if F2==1:
            aa = array([0,0,12,4,2])
            aa = [1/sqrt(a) if abs(a)>0 else 0 for a in aa]
        return aa[m1+2]
    
    if F1==1:
        if F2==2:
            aa = array([2,4,12])
            aa = [-1/sqrt(a) if abs(a)>0 else 0 for a in aa]
        if F2==1:
            aa = array([0,12,12])
            aa =  [1/sqrt(a) if abs(a)>0 else 0 for a in aa]
        return aa[m1+1]


# Pryecciones de los niveles

ind_a_num= [[1, 1],
            [1, 0],
            [1,-1],
            [2, 2],
            [2, 1],
            [2, 0],
            [2,-1],
            [2,-2],
            [3, 1],
            [3, 0],
            [3,-1]
            ]
D1_p = zeros((N,N))  # Para polarización σ+
D1_m = zeros((N,N))  # Para polarización σ+
D1_c = zeros((N,N))  # Para polarización σ+  en un haz y σ- en el otro


for ii,i1 in enumerate(ind_a_num):
    for jj,i2 in enumerate(ind_a_num):
        F1  = i1[0] if i1[0]<3 else i1[0]-2
        F2  = i2[0] if i2[0]<3 else i2[0]-2
        m1  = i1[1]
        m2  = i2[1]
        if (i1[0] in [1,2]) and (i2[0] in [3,4]):
            D1_p[jj,ii] = D1_p_fun(F1,m1,F2,m2)
            D1_m[jj,ii] = D1_m_fun(F1,m1,F2,m2)
            D1_c[jj,ii] = D1_p_fun(F1,m1,F2,m2) if F1==1 else D1_m_fun(F1,m1,F2,m2)
            
        elif (i1[0] in [3,4]) and (i2[0] in [1,2]):
            D1_p[jj,ii] = D1_p_fun(F2,m2,F1,m1)
            D1_m[jj,ii] = D1_m_fun(F2,m2,F1,m1)
            D1_c[jj,ii] = D1_p_fun(F2,m2,F1,m1) if F1==1 else D1_m_fun(F2,m2,F1,m1)
        else:
            D1_p[jj,ii] = 0
            D1_m[jj,ii] = 0
            D1_c[jj,ii] = 0




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Armado del sistema de ecuaciones usando SymPy
###############################################################################

"""
    Vamos a construir un modelo del tipo:
    dro/dt = (-i/hbar) [H,ro] + (1/2) * {S,ro} + L
    
    S es diagonal y tiene los decaimientos
    L es diagonal y tiene la repoblación de los niveles. L=L(ro)
    
    Para mantener la población, tr(S ro)-tr(L) == 0
    H es hermítico.  
    ro es hermítico y tr(ro) = 1
"""


from time import time
from numpy import zeros
import re

t0 = time()

from sympy import MatrixSymbol, Matrix, symbols, linear_eq_to_matrix, simplify
from sympy.parsing.sympy_parser import parse_expr
import sympy  as sy
from fractions import Fraction as fr




def s(m,n):
    """
    Representa la matriz/operador:
        |m><n|
    """
    if min(m,n)<0 or max(m,n)>N:
        raise ValueError(f'n,m deben estar entre 0 y {N}')
    rta = zeros((N,N)).astype(int)
    rta[m,n] = 1 
    return Matrix(rta)



ro =  Matrix( zeros((N,N)).astype(int) )

for ii in range(ro.shape[0]):
    for jj in range(ro.shape[1]):
        if ii==jj:
            ro[ii,jj] = symbols(f'ro{ii}_{jj}', real=True)
        else:
            ro[ii,jj] = symbols(f'ro{ii}_{jj}')

# Prueba: fuerzo hermiticidad
#for ii in range(ro.shape[0]):
#    for jj in range(ro.shape[1]):
#        if ii>jj:
#            ro[ii,jj] = sy.conjugate(ro[jj,ii])
# ro[N-1,N-1] = 1- Tr(ro) + ro[15,15]


Omega_A,Omega_B,Gamma,gamma1,gamma2  = symbols('Omega_A Omega_B Gamma gamma1 gamma2', real=True , positive=True)
Bz,Bt, Delta,delta = symbols('Bz Bt Delta delta', real=True)



H  = Matrix( zeros((N,N)).astype(int) , real=True)
S  = Matrix( zeros((N,N)).astype(int) , real=True)
L  = Matrix( zeros((N,N)).astype(int) , real=True)


# Matriz de los niveles de energía (en el interaction picture, solo los detunings)

for ii,de in enumerate( [Delta/2]*3+[-Delta/2]*5+[-delta]*3  ):
    H += de*s(ii,ii)



# Rearmo las matrices de D para cada polarización, pero usando simbólico

D1p = Matrix( zeros((N,N)))
D1m = Matrix( zeros((N,N)))
D1c = Matrix( zeros((N,N)))

for ii in range(D1p.shape[0]):
    for jj in range(D1p.shape[1]):
        val = D1_p[ii,jj]
        D1p[ii,jj] = sy.sqrt(fr(val**2).limit_denominator(20)) * int(sign(val))
        
        val = D1_m[ii,jj]
        D1m[ii,jj] = sy.sqrt(fr(val**2).limit_denominator(20)) * int(sign(val))
        
        val = D1_c[ii,jj]
        D1c[ii,jj] = sy.sqrt(fr(val**2).limit_denominator(20)) * int(sign(val))



#####################################
# Esto es: polarización lineal
D = (D1p + D1m)
# D = (D1p + D1m*parse_expr('1j'))
# Esto es polarizacion circular
# D = D1p
# D = D1m
#D = D1c

#
#####################################

# Le agrego a H las componentes de interacción dipolar D

for m in arange(3): 
    for n in arange(8,N) :
        H += Omega_A/2 *D[m,n]*s(m,n)
        H += Omega_A/2 *D[n,m]*s(n,m)

for m in arange(3,8): 
    for n in arange(8,N) :
        H += Omega_B/2 *D[m,n]*s(m,n)
        H += Omega_B/2 *D[n,m]*s(n,m)


A1,A2,A3,A4 = symbols('A1 A2 A3 A4', real=True)
Q0,Q1       = symbols('Q0 Q1', real=True)



H_B = Bz * (  A1*(   s(0 , 0) - s( 2, 2) ) +
              A2*( 2*s(3 , 3) + s( 4, 4) - s( 6, 6) - 2*s( 7, 7) ) +
              A3*(   s(8 , 8) - s(10,10) ) ) 


#H_B += Bz**2 * ( Q0 * (s(1,1)+s(5,5)              ) +
#                 Q1 * (s(0,0)+s(2,2)+s(4,4)+s(6,6))  )

H_B += Bz**2 * ( Q0 * (-s(1,1)+s(5,5)              ) +
                 Q1 * (-s(0,0)-s(2,2)+s(4,4)+s(6,6))  )


H_Bt = Bt * (  A1/sy.sqrt(2)*(   s(1 , 0) + s( 2, 1) ) +
               A2        *(   s(4 , 3) + sy.sqrt(6)/2*(s(5,4) + s(6,5)) + s( 7, 6) ) +
               A3/sy.sqrt(2)*(   s(9, 8) + s(10,9) )  )

H_B += H_Bt + H_Bt.T  # sy.conjugate(H_Bt.T)

H += H_B

S = Matrix( zeros((N,N)).astype(int)  )
for ii,g in enumerate([gamma1]*3+[gamma2]*5+[Gamma]*3) :
    S  +=   s(ii,ii)*g  



# Construyo la matriz de repoblamiento
L = Matrix( zeros((N,N)).astype(int)  )


for m in range(0,8):
    for n in range(8,N):
        #TOT = sum([ D[n,mp]**2 for mp in range(0,8)])
        TOT = sum([ D[n,mp]*sy.conjugate(D[n,mp]) for mp in range(0,8)])
        if abs(D[n,m])>0:
            if TOT == 0:
                raise ValueError('TOT == 0')
            L +=  Gamma * s(m,m) *  (D[n,m]**2)/TOT * ro[n,n]



for m in range(0,3):
    for n in range(3,8):
        L +=  gamma2* s(m,m)/3 * ro[n,n]

for m in range(3,8):
    for n in range(0,3):
        L +=  gamma1* s(m,m)/5 * ro[n,n]



#### Chequeo de armado correcto de L:
# La traza de (S*ro-L) debe ser CERO

if False:
    AA = S*ro - L
    sum( [ AA[ii,ii] for ii in range(N) ] )



###########################################################
Dro = -(H*ro - ro*H)*parse_expr('1j') - ( S*ro + ro*S )/2 + L
###########################################################



# Procesamos para obtener una expresión matricial vectorial


# Lista ordenada de elementos de matriz de ro
# Corresponden a la base vectorial sobre la que se escribirán 
# los vectores de cálculo
lista_de_simbolos = []
for ii in range(N):
    for jj in range(N):
        lista_de_simbolos += [ ro[ii,jj] ]

# Vamos a vectorizar la matriz del Hamiltoniano
# Cada elemento va a ser escrito como combinacion lineal los elementos de 
# matriz del operador densidad.
# Luego, lista_de_simbolos será la nueva base de descripción del problema algebráico


Mtxt = []
Ms = Matrix([[]])

for ii in range(N):
    for jj in range(N):
        print(f'M[{ii},{jj}]')
        
        # Definimos ecuación
        eqn = Dro[ii,jj]
        
        # Extraemos expresion de fila de matriz
        M, b = linear_eq_to_matrix(eqn, *lista_de_simbolos)
        
        # Convierto la expresión a texto y la almaceno para evaluarla luego en NumPy
        tmp =   str(M)[9:-3]
        Mtxt += [f'[{tmp}]']
        Ms = Ms.row_insert( 1, M )


del H,S,L,Dro,H_B,H_Bt,ro
del Omega_A,Omega_B,Delta,delta,Gamma,gamma1,gamma2,Bz,Bt


del A1,A2,A3,A4,Q0,Q1

tf = time()-t0
print(f'\n\n\nTime: {tf} seg | {round(tf/60,1)} min')




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Defino parametros y armo M
###############################################################################

I = 1j
A = array([ - 0.7022, 0.6994, - 0.2347, 0.2319 ])*1e6 #  MHz / G
Q = array([ 286.57, 215.68 ] )  #  Hz / G 2

A1,A2,A3,A4 = A
Q0,Q1       = Q

# Obtenemos la matriz de cada parámetro
# Dejamos en cada paso un parametro en 1 y el resto en zero.
# Guardamos las matrices para multipliar luego
Mpar = []
for par in eye(15).tolist():
    Omega_A,Omega_B,Delta,delta,Bz,Bt,Gamma,gamma1,gamma2,A1,A2,A3,A4,Q0,Q1 = par
    Mp = []
    for fila in Mtxt:
        Mp.append( eval(fila) )
    Mp = array(Mp)
    Mpar.append(Mp)

Gamma,gamma1,gamma2             = int(6e6) , 200 , 200  # Hz
Omega_A,Omega_B,Delta,delta,Bz,Bt = Gamma/60, Gamma/60, 0 , 0 , 0 , 0
A1,A2,A3,A4 = A
Q0,Q1       = Q

M = []
for fila in Mtxt:
    M.append( eval(fila) )

M = array(M)


parametros = Omega_A,Omega_B,Delta,delta,Bz,Bt,Gamma,gamma1,gamma2,A1,A2,A3,A4,Q0,Q1


Mb     = zeros((N**2,N**2)).astype(complex)
par_ii = arange(len(parametros)).tolist()

for ii in par_ii:
    Mb += parametros[ii] * Mpar[ii]

# Chequeo que M==Mb
abs(M-Mb).max()


Mb     = zeros((N**2,N**2)).astype(complex)
par_ii = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

for ii in par_ii:
    Mb += parametros[ii] * Mpar[ii]


   
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Evolución temporal del sistema dinámico
###############################################################################

def s(m,n):
    """
    Representa la matriz/operador:
        |m><n|
    Esta es la versión de NumPy
    """
    if min(m,n)<0 or max(m,n)>N*N:
        raise ValueError(f'n,m deben estar entre 0 y {N*N}')
    rta = zeros((N,N))
    rta[m,n] = 1 
    return rta
    

#S = Omega_c/sy.sqrt(Omega_c**2+Omega_p**2)
#C = Omega_p/sy.sqrt(Omega_c**2+Omega_p**2)
#U = s(0,1)*S+s(0,0)*C +s(1,1)*C - s(1,0)*S + 1*s(2,2)
#simplify(U*H0*U.inv())
#simplify(U*H1*U.inv())

# ver(AAi.dot(M).dot(AA))


def fun_vec(T,v0,delta=0,Delta=0,Omega_A=Gamma/60,Omega_B=Gamma/60 , Bz=0 , Bt=0, cambio_de_base=False):
    """
    Calcula la evolución temporal.
    Tenemos: dρ/dt = M·ρ
    Y      :     M = AA·D·AA^{-1}    con D diagonal
    Luego  :  ρ(t) = AA·exp(D t)·AA^{-1} · ρ_0
    
    fun_vec(T,v0):
        T : tiempo (puede ser un array)
        v0: ρ_0 (en su forma vectorial)
        
        devuelve: ρ(t) (en forma vectorial) 
    """
    
    #M = Mb + Delta*Mpar[2] + delta*Mpar[3] + Bz*Mpar[4]+Omega_A*Mpar[0] + Omega_B*Mpar[1]

    # Hago un cambio de base de M a su base diagonal
    # AA^{-1} · M · AA = D  que es diagonal
    
    M = []
    for fila in Mtxt:
        M.append( eval(fila) )
    
    M = array(M)
    
    aa, AA = linalg.eig(M)
    AAi    = linalg.inv(AA)
    
    
    # D = eye(M.shape[0])*exp(aa*t)
    if iterable(T):
        rta = array([ AA.dot(  eye(M.shape[0])*exp(aa*t)  ).dot(AAi).dot(v0) for t in T])
    else:
        rta = AA.dot(  eye(M.shape[0])*exp(aa*T)  ).dot(AAi).dot(v0)

    if cambio_de_base:
        S = Omega_A/sqrt(Omega_A**2+Omega_B**2)
        C = Omega_B/sqrt(Omega_A**2+Omega_B**2)
        U = s(0,1)*S+s(0,0)*C +s(1,1)*C - s(1,0)*S + 1*s(2,2)
        Ui= linalg.inv(U)
        #simplify(U*H0*U.inv())
        #simplify(U*H1*U.inv())
        if iterable(T):
            rta = array([ ( U.dot(r.reshape(N,N)).dot(Ui) ).flatten() for r in rta ] )
        else:
            rta = ( U.dot(rta.reshape(N,N)).dot(Ui) ).flatten()
            
    
    return rta

## Estado inicial fictiocio
#ro0      = zeros((N,N))
#ro0[0,0] = 1
#vec0     = ro0.flatten()





# Para una intensidad similar a la intesidad de saturación tenemos 
# Omega_sat = sqrt(2*30*4*pi/hbar)*cef**2 *5*a0
# Omega_sat ~ 37 hKhz

#######################################################
# Estado inicial: Construyo un estado térmico.
# Para eso tengo que tener un vector de las energías (sin campo mag)
Kboltz = 1.380649e-23       # J / K
hbar   = 1.054571817e-34    #J s
hplank = 6.62607015e-34     #J s

E_vec  = array(  [0]*3 + [6.4]*5 + [377107-0.5]*3  ) # en GHz
Prob   = exp(-hplank*E_vec*1e9/(300*Kboltz)) / sum( exp(-hplank*E_vec*1e9/(300*Kboltz)) )
ro0    = eye(N) * Prob
vec0   = ro0.flatten()
#######################################################



t0   = time()

# Régimen de alta intensidad
t    = linspace(0,1e-6,1000)
t    = logspace(-6,2,100)

#vec2 = fun_vec(t,vec0, delta=10e3, Delta=0, Omega_A=30e6,Omega_B=30e6, cambio_de_base=False )
vec2 = fun_vec(t,vec0, delta=10e3, Delta=0, Omega_A=Gamma/60,Omega_B=Gamma/60,Bz=30, cambio_de_base=False )

print(time()-t0)

## Régimen de baja intensidad
#t    = linspace(0,0.2,10000)
#vec2 = fun_vec(t,vec0, delta=10e3, Delta=0, Omega_A=30e3,Omega_B=30e3 , cambio_de_base=False )


# ro2  = vec2[-1].reshape(N,N)


# Graficamos resultados -------------------------------------------------------
indices = arange(N**2).reshape(N,N)

#fig, axx = plt.subplots(2,1, figsize=(6,5),  constrained_layout=True , sharex=True)
diagonal = diag(indices).tolist()


if False:

    fig, ax = plt.subplots(1,1 , figsize=(17,7) , sharey=True )
    
    ro = array([ vv.reshape(N,N) for vv in vec2 ])
    
    
    colores = ['C0']*3 + ['C1']*5 + ['C2']*3
    for jj,ii in enumerate(diagonal):
        ax.plot( t, abs(vec2[:,ii]) , label=r'$\rho_{'+f'{jj},{jj}'+r'}$' , color=colores[jj])
        #ax.plot( t, ro[:,jj,jj] , label=r'$\rho_{'+f'{jj},{jj}'+r'}$' , color=colores[jj])
        print(jj)
        ax.text( t[-1] +50*jj , abs(vec2[-1,ii]), r'$\rho_{'+f'{jj},{jj}'+r'}$'  , color=colores[jj], fontsize=14 )
    
    ax.grid(b=True,linestyle='--',color='lightgray')
    ax.semilogx()
    


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Barrido en el detuning de dos fotones Delta
###############################################################################

t0 = time()

t    = 3  # seg

ancho = 0.004
largo = 50
Deltas  = array( sorted(logspace(1,log10(Gamma*ancho),largo).tolist() + (-logspace(1,log10(Gamma*ancho),largo)).tolist() ) )
Deltas += 0  # para Bz = 100/2/pi

tmp    = array(sorted( Deltas.tolist() +  (Deltas+14e3).tolist() + (Deltas-14e3).tolist() ))
Deltas = tmp



#Deltas  = linspace(-Gamma*1,Gamma*1,100)

OmegaL = Gamma/30


vec2 = array([ fun_vec(t,vec0, delta=0, Delta=Delta, Omega_A=OmegaL,Omega_B=OmegaL, 
                       Bz=0.0, Bt=0.0,
                       cambio_de_base=False ) for Delta in Deltas ])



tf = time() - t0 

print(f'\n\n\nTime: {tf} seg | {round(tf/60,1)} min')



if __name__ == "__main__":
    
    fig, axx = plt.subplots(4,1, figsize=(10,8),  constrained_layout=True , sharex=True)
    
    indices = arange(N**2).reshape(N,N)
    diagonal = diag(indices).tolist()
    
    ff = Deltas/1e3
    
    ax = axx[0]
    for ii,jj in enumerate(diagonal[:3]):
        ax.plot( ff , abs(vec2[:,jj]) , label=f'$\\rho_{{{ii},{ii}}}$', alpha=0.7)
    ax.legend(ncol=2)
    ax.set_title('Estados fundamentales F=1', pad=-15, loc='left')
    
    ax = axx[1]
    for ii,jj in enumerate(diagonal[3:8]):
        ii += 3
        ax.plot( ff , abs(vec2[:,jj]) , label=f'$\\rho_{{{ii},{ii}}}$', alpha=0.7)
    ax.legend(ncol=2)
    ax.set_title('Estados fundamentales F=2', pad=-15, loc='left')
    
    
    ax = axx[2]
    for ii,jj in enumerate(diagonal[8:]):
        ii += 8
        ax.plot( ff , abs(vec2[:,jj]) , label=f'$\\rho_{{{ii},{ii}}}$', alpha=0.7)
    ax.legend(ncol=2)
    ax.set_title('Estados excitados F=1', pad=-15, loc='left')
    
    
    ax = axx[3]
    absorbancia = ff*0
    #for ii,jj in zip([7,5,1],[10,10,10]):
    #for ii,jj in zip([0,9,4,6],[9,2,9,9]):
    #for ii,jj in zip([3,5,1],[8,8,8]):
    for ii,jj in zip([3,5,1,0,9,4,6,7,5,1],[8,8,8,9,2,9,9,10,10,10]):
    
        kk           = indices[ii,jj]
        absorbancia += imag(vec2[:,kk])
        #ax.plot( ff, imag(vec2[:,kk]) )
    Tsalida = exp(-absorbancia/(Gamma/3)*10e6)
    ax.plot( ff , Tsalida-min(Tsalida) , color='black')
    
    ax.set_ylabel('Transmitancia')
    
    
    ax.set_xlabel(r'$\Delta$ [kHz]')
    
    
    fig.suptitle(r'CPT con $\Omega_A=\Omega_A=\Gamma/3 , Bz=0$',weight='bold')
    fig.savefig('modelo_cpt_11_niveles_01.png')


#%%

vec2 = array([ fun_vec(t,vec0, delta=0, Delta=Delta, Omega_A=OmegaL,Omega_B=OmegaL, 
                       Bz=0.01, Bt=0.0,
                       cambio_de_base=False ) for Delta in Deltas ])


if __name__ == "__main__":
    
    fig, axx = plt.subplots(4,1, figsize=(10,8),  constrained_layout=True , sharex=True)
    
    indices = arange(N**2).reshape(N,N)
    diagonal = diag(indices).tolist()
    
    ff = Deltas/1e3
    
    ax = axx[0]
    for ii,jj in enumerate(diagonal[:3]):
        ax.plot( ff , abs(vec2[:,jj]) , label=f'$\\rho_{{{ii},{ii}}}$', alpha=0.7)
    ax.legend(ncol=2)
    ax.set_title('Estados fundamentales F=1', pad=-15, loc='left')
    
    ax = axx[1]
    for ii,jj in enumerate(diagonal[3:8]):
        ii += 3
        ax.plot( ff , abs(vec2[:,jj]) , label=f'$\\rho_{{{ii},{ii}}}$', alpha=0.7)
    ax.legend(ncol=2)
    ax.set_title('Estados fundamentales F=2', pad=-15, loc='left')
    
    
    ax = axx[2]
    for ii,jj in enumerate(diagonal[8:]):
        ii += 8
        ax.plot( ff , abs(vec2[:,jj]) , label=f'$\\rho_{{{ii},{ii}}}$', alpha=0.7)
    ax.legend(ncol=2)
    ax.set_title('Estados excitados F=1', pad=-15, loc='left')
    
    
    ax = axx[3]
    absorbancia = ff*0
    #for ii,jj in zip([7,5,1],[10,10,10]):
    #for ii,jj in zip([0,9,4,6],[9,2,9,9]):
    #for ii,jj in zip([3,5,1],[8,8,8]):
    for ii,jj in zip([3,5,1,0,9,4,6,7,5,1],[8,8,8,9,2,9,9,10,10,10]):
    
        kk           = indices[ii,jj]
        absorbancia += imag(vec2[:,kk])
        #ax.plot( ff, imag(vec2[:,kk]) )
    Tsalida = exp(-absorbancia/(Gamma/3)*10e6)
    ax.plot( ff , Tsalida-min(Tsalida) , color='black')
    
    ax.set_ylabel('Transmitancia')
    
    
    ax.set_xlabel(r'$\Delta$ [kHz]')
    
    
    fig.suptitle(r'CPT con $\Omega_A=\Omega_A=\Gamma/3 , Bz=0.01$ G',weight='bold')
    fig.savefig('modelo_cpt_11_niveles_02.png')

