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


N = 3  # Número de niveles

# Función para construcción de operadores
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


# Contruyo la matriz del operador densidad ρ

#ro = Matrix(MatrixSymbol('ro', N, N))

ro =  Matrix( zeros((N,N)).astype(int) )

for ii in range(ro.shape[0]):
    for jj in range(ro.shape[1]):
        if ii==jj:
            # Las diagonales son reales positivas
            ro[ii,jj] = symbols(f'rho{ii}_{jj}', real=True, positive=True)
        else:
            ro[ii,jj] = symbols(f'rho{ii}_{jj}')


# Definimos parámetros del sistema
Omega_A,Omega_B,Gamma = symbols('Omega_A Omega_B Gamma', real=True , positive=True)
Delta,delta         = symbols('Delta delta', real=True)


H  = Matrix( zeros((N,N)))
S  = Matrix( zeros((N,N)))
L  = Matrix( zeros((N,N)))


# D = 1*s(0,2) + 1*s(1,2) + 1*s(2,0) + 1*s(2,1)


H  = s(0,0)*Delta - s(1,1)*Delta -2*s(2,2)*delta
H += s(0,2)*Omega_A + s(1,2)*Omega_B
H += s(2,0)*Omega_A + s(2,1)*Omega_B
H /= 2

S = s(2,2) * Gamma
L = (s(0,0) + s(1,1))* Gamma/2 * ro[2,2]


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


del H,S,L,Dro
del Omega_A,Omega_B,Delta,delta,Gamma


tf = time()-t0
print(f'\n\n\nTime: {tf} seg | {round(tf/60,1)} min')




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Defino parametros y armo M
###############################################################################


parametros                        = 6e6/60, 6e6/60, 0 , 10e3 , 6e6
Omega_A,Omega_B,Delta,delta,Gamma = parametros

I = 1j

# Como todos los parámetros son lineales, vamos a separar M en partes Mpar[j]
# tal que podamos fabricar fácilmente la matriz de cada condición experimental
# sumando param[j]*Mpar[j]

Mpar = []

for par in eye(5).tolist():
    Omega_A,Omega_B,Delta,delta,Gamma = par
    Mp = []
    for fila in Mtxt:
        Mp.append( eval(fila) )
    Mp = array(Mp)
    Mpar.append(Mp)



Omega_A,Omega_B,Delta,delta,Gamma = parametros
M = Omega_A*Mpar[0] + Omega_B*Mpar[1] + Delta*Mpar[2] + delta*Mpar[3] + Gamma*Mpar[4]

   
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


def fun_vec(T,v0,delta=0,Delta=0,Omega_A=6e6/60,Omega_B=6e6/60 , cambio_de_base=False):
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
    Gamma = 6e6*2*pi

    M = Omega_A*Mpar[0] + Omega_B*Mpar[1] + Delta*Mpar[2] + delta*Mpar[3] + Gamma*Mpar[4]
    
    # Hago un cambio de base de M a su base diagonal
    # AA^{-1} · M · AA = D  que es diagonal
    
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

# Estado inicial térmico ------------------------------------------------------
Energias       = array([0,6.8e9,377e12])*hbar
Temperatura    = 273.15+20
Probabilidades = exp(- Energias/(Kmks*Temperatura))

ro0      = zeros((N,N))
for jj in range(3):
    ro0[jj,jj] = Probabilidades[jj]/sum(Probabilidades)
vec0     = ro0.flatten()


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Evolución temporal de poblaciones para altan intensidad de haz
###############################################################################


t0   = time()

# Régimen de alta intensidad
t    = linspace(0,0.5e-6,10000)
vec2 = fun_vec(t,vec0, delta=10e3, Delta=0, Omega_A=Gamma*50,Omega_B=Gamma*50, cambio_de_base=False )


# Régimen de baja intensidad
#t    = linspace(0,0.2,10000)
#vec2 = fun_vec(t,vec0, delta=10e3, Delta=0, Omega_A=30e3,Omega_B=30e3 , cambio_de_base=False )

ro2  = vec2[-1].reshape(N,N)


# Graficamos resultados -------------------------------------------------------
indices  = arange(9).reshape(3,3)
diagonal = diag(indices).tolist()




fig, axx = plt.subplots(3,1, figsize=(6,5),  constrained_layout=True , sharex=True)

ax=axx[0]
for ii,nombre in zip(diagonal[::2],r'$\rho_{00}$ y $\rho_{11}$,$\rho_{22}$'.split(',')):
    ax.plot( t*1e6 , abs(vec2[:,ii]) , label=nombre)

ax.set_ylabel('Poblaciones')

ax=axx[1]
for ii,nombre in zip([1,2,5],r'$|\rho_{01}|$ $|\rho_{02}|$ $|\rho_{12}|$'.split()):
    ax.plot( t*1e6 , abs(vec2[:,ii]) , label=nombre)

ax.set_ylabel('Superposición')


ax.legend()

ax=axx[2]

pureza = array([ Tr(abs(vv.reshape(N,N).dot(vv.reshape(N,N)))) for vv in vec2 ])
ax.set_ylabel(r'Pureza Tr$(\rho^2)$')

ax.plot( t*1e6 , pureza )

vec2 = fun_vec(t,vec0, delta=10e3, Delta=0, Omega_A=Gamma*50,Omega_B=Gamma*50, cambio_de_base=True )

ax=axx[0]
ax.plot( t*1e6 , abs(vec2[:,4]) , label=r'$\rho_{DD}$')
ax.plot( t*1e6 , abs(vec2[:,0]) , label=r'$\rho_{BB}$')

ax.legend()


for ax in axx:
    ax.grid(b=True,linestyle='--',color='lightgray')
ax.set_xlabel('tiempo [us]')






# fig.savefig('modelo_cpt_3_niveles_01.png')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Barrido en el detuning Raman
###############################################################################

#vec2 = array([ fun_vec(t,vec0, delta=10e3, Delta=Delta, Omega_A=300e6,Omega_B=300e6 , 
#                       cambio_de_base=False ) for Delta in Deltas ])
#



t    = 1
Deltas = linspace(-10*Gamma,10*Gamma,10001)
vec2 = array([ fun_vec(t,vec0, delta=1e3, Delta=Delta, Omega_A=Gamma*2,Omega_B=Gamma*2 , 
                       cambio_de_base=True ) for Delta in Deltas ])


# Graficamos resultados -------------------------------------------------------
indices = arange(9).reshape(3,3)

fig, axx = plt.subplots(2,1, figsize=(6,5),  constrained_layout=True , sharex=True)
diagonal = diag(indices).tolist()

ax=axx[0]
for ii,nombre in zip(diagonal,r'$\rho_{BB}$ $\rho_{DD}$ $\rho_{22}$'.split()):
    ax.plot( Deltas/1e6 , abs(vec2[:,ii]) , label=nombre)
ax.legend()


ax=axx[1]
#for ii,nombre in zip([1,2,5],r'$|\rho_{01}|$ $|\rho_{02}|$ $|\rho_{12}|$'.split()):
#    ax.plot( Deltas , abs(vec2[:,ii]) , label=nombre)
#ax.legend()


pureza = array([ Tr(abs(vv.reshape(N,N).dot(vv.reshape(N,N)))) for vv in vec2 ])
ax.plot( Deltas/1e6 , pureza )

ax.set_ylabel(r'Pureza Tr$(\rho^2)$')

for ax in axx:
    ax.grid(b=True,linestyle='--',color='lightgray')
ax.set_xlabel('$\\Delta$ [MHz]')

fig.suptitle('Poblaciones y pureza para barrido en $\\Delta$')

# fig.savefig('modelo_cpt_3_niveles_02.png')


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Barrido en el detuning óptico, comparado con y sin CPT
###############################################################################




#
## Régimen de alta intensidad
#t    = 1
#deltas = linspace(-10*Gamma,10*Gamma,10001)
#vec2 = array([ fun_vec(t,vec0, delta=delta, Delta=0.3*Gamma, Omega_A=Gamma,Omega_B=Gamma , 
#                       cambio_de_base=True ) for delta in deltas ])
#
##vec2 = array([ fun_vec(t,vec0, delta=10e3, Delta=Delta, Omega_A=300e6,Omega_B=300e6 , 
##                       cambio_de_base=False ) for Delta in Deltas ])
##
#
#
## Graficamos resultados -------------------------------------------------------
#indices = arange(9).reshape(3,3)
#
#fig, ax = plt.subplots(1,1, figsize=(6,5),  constrained_layout=True , sharex=True)
#diagonal = diag(indices).tolist()
#
#for ii,nombre in zip(diagonal,r'$\rho_{00}$ $\rho_{11}$ $\rho_{22}$'.split()):
#    ax.plot( deltas , abs(vec2[:,ii]) , label=nombre)
#
#
#
#
#vec2 = array([ fun_vec(t,vec0, delta=delta, Delta=0.3*Gamma, Omega_A=Gamma,Omega_B=Gamma*0 , 
#                       cambio_de_base=True ) for delta in deltas ])
#
#for ii,nombre in zip(diagonal,r"$\rho_{00}'$ $\rho_{11}'$ $\rho_{22}'$".split()):
#    ax.plot( deltas , abs(vec2[:,ii]) ,':', label=nombre , color=f'C{int(ii//4)}')
#
#
#
#
#ax.legend()
#
#ax.grid(b=True,linestyle='--',color='lightgray')
#ax.set_xlabel('$\\delta$ [Hz]')



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Variación de la absorción por CPT
###############################################################################


# Del Loudon ( The Quantum Theory of Light - 3rd ed Oxford Science Publications )
# podemos ver que la polarización de un átomo es proporcional a 
# La susceptibilidad chi. 
# Por otro lado, la absorbancia K(w) = (w/c) imag(chi) / n
# con n el índice de refraxion ~ 1
# Por ende la absorbancia es proporcional a imag(P)
# P = (N/V) <psi| x |psi> = (N/V) Tr(rho * d)
# Con ello podemos simular una respuesta proporcional a la tranmisntacia

 



# Barro en intensidades:

fig, axx = plt.subplots(1,2, figsize=(6,5),  constrained_layout=True , sharey=True)
diagonal = diag(indices).tolist()
indices = arange(9).reshape(3,3)

t    = 1
#Deltas = linspace(-10*Gamma,10*Gamma,1001)

ancho  = 5
largo  = 500
Deltas = array( sorted(logspace(1,log10(Gamma*ancho),largo).tolist() + (-logspace(1,log10(Gamma*ancho),largo)).tolist() ) )


for aa in [ 0.2,  0.4,  0.8,  1.6, 3.2]:
    
    II = aa * Gamma
    vec2 = array([ fun_vec(t,vec0, delta=0, Delta=Delta, Omega_A=II,Omega_B=II , 
                           cambio_de_base=True ) for Delta in Deltas ])

    Tsalida =  exp(  -imag( (vec2[:,2]+vec2[:,5])*1e7/II )  )
    
    axx[0].plot( Deltas/1e6 ,Tsalida , label=f'{aa}' , alpha=0.6)
    axx[1].plot( Deltas[Deltas>0] ,Tsalida[Deltas>0] , label=f'{aa}')


axx[0].legend()
axx[0].set_xlabel('$\\Delta$ [MHz]')
axx[0].set_ylabel('Transmitancia')

axx[1].set_xlabel('$\\Delta$ [Hz]')

for ax in axx:
    ax.grid(b=True,linestyle='--',color='lightgray')

ax.semilogx()

fig.suptitle('Transmitancia CPT para diferentes intensidades de haz')


# fig.savefig('modelo_cpt_3_niveles_03.png')



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Variación doble de la absorción por CPT
###############################################################################




# Barro en intensidades:

diagonal = diag(indices).tolist()
indices = arange(9).reshape(3,3)

t    = 1
#Deltas = linspace(-10*Gamma,10*Gamma,1001)

ancho  = 10
largo  = 100
deltas = array( sorted(logspace(1,log10(Gamma*ancho),largo).tolist() + (-logspace(1,log10(Gamma*ancho),largo)).tolist() ) )

ancho  = 1
largo  = 120
Deltas = array( sorted(logspace(1,log10(Gamma*ancho),largo).tolist() + (-logspace(1,log10(Gamma*ancho),largo)).tolist() ) )


aa = 0.5    
II = aa * Gamma

mapa = []
for delta in deltas:
    vec2 = array([ fun_vec(t,vec0, delta=delta, Delta=Delta, Omega_A=II,Omega_B=II , 
                           cambio_de_base=True ) for Delta in Deltas ])
    
    Tsalida =  exp(  -imag( (vec2[:,2]+vec2[:,5])*1e7/II )  )
    mapa.append( Tsalida.tolist() )


#ax.plot( Deltas/1e6 ,Tsalida , label=f'{aa}' , alpha=0.6)

mapa = array(mapa)



fig, ax = plt.subplots(1,1, figsize=(6,5),  constrained_layout=True , sharey=True)

ax.pcolormesh(Deltas/1e6, deltas/1e6,mapa)
ax.colorbar()

ax.legend()
ax.set_xlabel('$\\Delta$ [MHz]')
ax.set_ylabel('$\\delta$ [MHz]')

ax.set_xlabel('$\\Delta$ [Hz]')

ax.grid(b=True,linestyle='--',color='lightgray')


#fig.suptitle('Transmitancia CPT para diferentes intensidades de haz')


# fig.savefig('modelo_cpt_3_niveles_03.png')


#%%

from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 

from matplotlib.colors import LightSource

z = mapa

x, y = meshgrid(Deltas/1e6, deltas/1e6)

# Set up plot
fig, ax = plt.subplots(subplot_kw=dict(projection='3d'),  constrained_layout=True)

ax.view_init(39,-60)

ls = LightSource(270, 45)
# To use a custom hillshading mode, override the built-in shading and pass
# in the rgb colors of the shaded surface calculated from "shade".
rgb = ls.shade(z, cmap=plt.cm.gist_earth, vert_exag=0.1, blend_mode='soft')
surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=rgb,
                       linewidth=0, antialiased=False, shade=False, alpha=0.5)

plt.show()



cset = ax.contour(x, y, z, zdir='z', offset=0.8, cmap=plt.cm.coolwarm)
#cset = ax.contour(x, y, z, zdir='x', offset=-4, cmap=plt.cm.coolwarm)
#cset = ax.contour(x, y, z, zdir='y', offset=-60, cmap=plt.cm.coolwarm)

#ax.set_xlim(-40, 40)
#ax.set_ylim(-40, 40)
#ax.set_zlim(-100, 100)

ax.set_xlabel('$\\Delta$ [MHz]')
ax.set_ylabel('$\\delta$ [MHz]')
ax.set_zlabel('T')