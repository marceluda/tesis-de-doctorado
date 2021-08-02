# -*- coding: utf-8 -*-
"""
Modelo para graficar espectros de absorción de Rubidio
"""

#%% Librerías

from numpy import * 
import numpy as np
from matplotlib import pyplot as plt

import os



#%% Funciones para cálculo de transiciones 


def CG(J,M,j1,j2,m1,m2):
    """
    Cálculo de los CLEBSCH-GORDAN  <j1,j2;m1,m2|j1,j2;J,M>
    Definidos según el algoritmo de: https://en.wikipedia.org/wiki/Table_of_Clebsch%E2%80%93Gordan_coefficients
    Chequeado con las tablas del Griffiths
    """
    if J>j1+j2 or J<abs(j1-j2):
        raise ValueError(f'J,j1,j2 deben cumplir:  |j1-j2|<=J<=j1+j2\nJ={J},j1={j1},j2={j2}')
    
    #if not ( J>0 and (j1>0 or j2>0)):
    #    return 0
        #raise ValueError('Los coeficientes deben J,j1,j2 deben ser mayores a cero')
    
    if not M==m1+m2:
        return 0
    
    if not all( [ int(2*x) == round(2*x,2) for x in (J,M,j1,j2,m1,m2)] ):
        raise ValueError('Los parámetros deben ser enteros o semienteros')
    
    factorial = lambda x: math.factorial(int(x)) if int(x)==round(x,2) else False

    factores = lambda k: [k,j1+j2-J-k,j1-m1-k,j2+m2-k,J-j2+m1+k,J-j1-m2+k]
    
    p1 = (2*J+1) * factorial(J+j1-j2)*factorial(J+j2-j1)*factorial(-J+j1+j2)/factorial(J+j1+j2+1)
    
    p2 = factorial(J+M)*factorial(J-M)*factorial(j1+m1)*factorial(j1-m1)*factorial(j2+m2)*factorial(j2-m2)
    
    p3 = sum([ (-1)**k / product([ factorial(y) for y in factores(k) ]) for k in range(int(-J-j1-j2),int(J+j1+j2+1)) if all([ y>=0 for y in factores(k) ]) ])

    return sqrt(p1*p2)*p3




def W3j(A):
    """
    Calucla los coeficientes de Wigner 3-j:
        ( j1 , j2 , j3 )
        ( m1 , m2 , m3 )
    Referencia: https://en.wikipedia.org/wiki/3-j_symbol
    """
    A = array(A)
    
    if not A.shape == (2,3):
        raise ValueError('El argumento debe tener un shape==(2,3)')
    j1,j2,j3 = A[0]
    m1,m2,m3 = A[1]
    
    # Prueba:   W3j([[1,2,3],[0,1,-1]])             debería ser    sqrt(2/105)*2
    #           W3j([[1,1/2,3/2],[1,-1/2,-1/2]])    deberia ser   -sqrt(1/3)/2
    #           W3j([[1,2,3],[-1,-2,3]])            deberia ser   sqrt(1/7)
    
    # Chequeado con http://www-stone.ch.cam.ac.uk/cgi-bin/wigner.cgi?symbol=3j&j1=1&j2=2&j3=3&m1=-1&m2=-2&m3=3
    for jj,mm in zip(A[0],A[1]):
        #   -j <= m <= j
        if not round(mm,2) in arange(-jj,jj+1).round(2):
            #print('no cumple mm E +-jj')
            return 0
    # m1+m2+m3 == 0
    if not round(sum(A[1]),2) == 0:
        #print('no cumple sum mm ==0')
        return 0
    
    # |j1-j2| <= j3 <= j1+j2
    if not ( abs(j1-j2)<= j3 and j3<= j1+j2  ):
        #print('no cumple jj E  j-j , jj')
        return 0
    
    # Chequeamos reglas de selección:
    return (-1)**int(round(j1-j2-m3)) *1/sqrt(2*j3+1) * CG(j3,-m3,j1,j2,m1,m2)


def W6j(A):
    """
    Calucla los coeficientes de Wigner 6-j:
        { j1 , j2 , j3 }
        { j4 , j5 , j6 }
    Referencia: https://en.wikipedia.org/wiki/6-j_symbol
    """
    A = array(A)
    rta = 0
    
    j1,j2,j3 = A[0]
    j4,j5,j6 = A[1]
    
    #print(j1,j2,j3,j4,j5,j6)
    # Poco elegante, pero efectivo
    for m1 in arange(-j1,j1+1):
        for m2 in arange(-j2,j2+1):
            for m3 in arange(-j3,j3+1):
                for m4 in arange(-j4,j4+1):
                    for m5 in arange(-j5,j5+1):
                        for m6 in arange(-j6,j6+1):
                            
                            tmp  = (-1)**sum( array([j1,j2,j3,j4,j5,j6])-array([m1,m2,m3,m4,m5,m6]) )
                            tmp  = int(round(tmp))
                            
                            tmp *= W3j([[ j1, j2, j3],
                                        [-m1,-m2,-m3]])
    
                            tmp *= W3j([[ j1, j5, j6],
                                        [ m1,-m5, m6]])
    
                            tmp *= W3j([[ j4, j2, j6],
                                        [ m4, m2,-m6]])
    
                            tmp *= W3j([[ j4, j5, j3],
                                        [-m4, m5, m3]])
                            
                            rta += tmp
    return rta



def cmf(Fg,Jg,Lg,mFg,Fe,Je,Le,mFe,q=0,II=5/2,S=1/2):
    """
    Coeficientes que acompañan al operador dipolar de estructura fina 
    para definir los operadores de la estructura hiperfina.
    Tomado de Siddons (4): https://doi.org/10.1088/0953-4075/41/15/155004 
    """
    rta  = (-1)**(2*Fe+II+Jg+Je+Lg+S+mFg+1)
    
    rta *= sqrt( (2*Fg+1)*(2*Fe+1)*(2*Jg+1)*(2*Je+1)*(2*Lg+1)  )

    rta *= W3j([[  Fe,  1,   Fg],
                [ mFe, -q, -mFg]])
    
    rta *= W6j([[  Jg, Je,   1],
                [  Fe, Fg,  II]])
    
    rta *= W6j([[  Lg, Le,   1],
                [  Je, Jg,   S]])
    return rta


def Cf2(Fg,Jg,Lg,Fe,Je,Le,q=0,II=5/2,S=1/2):
    """
    Suma de coeficientes cmf al cuadrado sobre todos las proyecciones m
    para una dada transición.
    """
    rta = 0
    for mm in range(-Fg,Fg+1):
        rta += cmf(Fg,Jg,Lg,mm,Fe,Je,Le,mm+q,q,II,S)**2
        
    return rta


#%% Auxiliar

#    Estas funciones auxiliares permiten escribir en modo de fracciones 
#    las tablas de valores de Cf2 para compararlas con la de los
#    trabajos usados como referencia
#    Siddons https://doi.org/10.1088/0953-4075/41/15/155004


Chequear_valores = False

if Chequear_valores:
    from decimal import Decimal as D
    from fractions import Fraction as fr
    
    def adivinar_forma(x):
        """
        Infiere la forma fraccional de un número
        """
        
        if x==0:
            return '0'
        
        opciones = [ (a,b, fr(abs(x)**a*pi**b).limit_denominator() ) for a in [-2,-1,1,2] for b in range(-2,3)  ]
        
        # Me quedo con la de menos denominador
        rta      = list(sorted(opciones , key=lambda x: (x[2].denominator,x[2].numerator)))[0]
        
        a,b,rta_frac   =  rta 
        signo = '-' if sign(x)==-1 else '+'
        frac  = str(rta_frac) if a>0 else str(1/rta_frac)
        pival = '' if b==0 else '*pi' if b>0 else '/pi'
        if abs(a)==2:
            if abs(b)==1:
                return f'{signo}sqrt({frac}{pival})'
            else:
                return f'{signo}sqrt({frac}){pival}'
        else:
            return f'{signo}{frac}{pival}' 
        
    
    # Chequeo si me da lo mismo que Siddons, tabla Ba apendice B:
    for Fg in [2,3]:
        for Fe in [1,2,3,4]:
            valor = Cf2(Fg=Fg,Jg=1/2,Lg=0,Fe=Fe,Je=3/2,Le=1,q=0,II=5/2,S=1/2)
            print(adivinar_forma(valor), end='   ')
        print('\n'+'-'*30)
    
    #    +1/3   +35/81   +28/81   0   
    #    0      +10/81   +35/81   +sqrt(1)   
    
    
    for Fg in [1,2]:
        for Fe in [0,1,2,3]:
            valor = Cf2(Fg=Fg,Jg=1/2,Lg=0,Fe=Fe,Je=3/2,Le=1,q=0,II=3/2,S=1/2)
            print(adivinar_forma(valor), end='   ')
        print('\n'+'-'*30)
        
        
    #    +1/9   +5/18   +5/18   0   
    #    0      +1/18   +5/18   +7/9   
        
    for Fg in [2,3]:
        for Fe in [2,3]:
            valor = Cf2(Fg=Fg,Jg=1/2,Lg=0,Fe=Fe,Je=1/2,Le=1,q=0,II=5/2,S=1/2)
            print(adivinar_forma(valor), end='   ')
        print('\n'+'-'*30)
    


#%% Datos de la estrcutura


import json


#    Los valores de esta tabla fueron tomados de:
#    Steck http://steck.us/alkalidata
#    
#    Los valores efectivos de momento dipolar fueron tomandos de:
#    Siddons https://doi.org/10.1088/0953-4075/41/15/155004


# Si no existe en archivo de valores calculados, lo creamos

if not os.path.isfile('Rb.json'):
    
    atom = { 'name'   : 'Rubidium',
             'symbol' : 'Rb' ,
             'level'  : '5²' ,
             'isotopes': { 85: {} , 87: {} }
             }
    
    
    a85 = atom['isotopes'][85]
    a87 = atom['isotopes'][87]
    
    
    a85['levels'] = {}
    a85['levels']['S1/2']=  { 2: -1770.8439228 ,  # MHz
                              3: +1264.8885163 }  # MHz
    a85['levels']['P3/2'] = { 1:  -113.208  ,  # MHz
                              2:   -83.835  ,  # MHz
                              3:   -20.435  ,  # MHz
                              4:  +100.205  }  # MHz
    a85['levels']['P1/2'] = { 2:  -210.923  ,  # MHz
                              3:  +150.659  }  # MHz
    
    a85['mass'] = 1.409993199e-25 # Kg
    a85['I'   ] = 5/2 
    
    a85['D1'] = { 'wl'  : 794.979014933 , # nm
                  'f'   : 377.107385690 , # THz 
                  'E'   : 1.559590695   , # eV 
                  'link': ['S1/2','P1/2'] ,
                  #'S'   : { 2:{ 2: 10/81 , 3: 35/81 } , 3: { 2: 35/81, 3: 28/81 } }, # Sacado del siddons
                  'DecayRate': 36.129e6  , # Hz
                  'DipoleMoment': 5.182  # e a0
                  }
    
    a85['D2'] = { 'wl'  : 780.241368271  , # nm
                  'f'   : 384.230406373  , # THz 
                  'E'   : 1.589049139    , # eV 
                  'link': ['S1/2','P3/2'],
                  #'S'   : { 2:{ 1: 1/3 , 2: 35/81, 3:28/81, 4:0 } , 3: { 1: 0 , 2: 10/81, 3:35/81, 4:1 } }, # Sacado del siddons
                  'DecayRate': 38.117e6 ,# Hz
                  'DipoleMoment': 5.177  # e a0
                  }
    
    
    a87['levels'] = {}
    a87['levels']['S1/2']=  { 1: -4271.676631815181 ,  # MHz
                              2: +2563.00597908910 }  # MHz
    a87['levels']['P3/2'] = { 0:  -302.0738  ,  # MHz
                              1:  -229.8518  ,  # MHz
                              2:   -72.9112  ,  # MHz
                              3:  +193.7407  }  # MHz
    a87['levels']['P1/2'] = { 1:  -509.05  ,  # MHz
                              2:  +305.43  }  # MHz
    a87['mass'] = 1.443160648e-25 # Kg
    a87['I'   ] = 3/2 
    
    a87['D1'] = { 'wl'  : 794.978851156 , # nm
                  'f'   : 377.107463380 , # THz 
                  'E'   : 1.559591016   , # eV 
                  'link': ['S1/2','P1/2'] ,
                  #'S'   : { 1:{ 1: 1/18 , 2: 5/18 } , 2: { 1: 5/18, 2: 5/18 } }, # Sacado del siddons
                  'DecayRate': 36.129e6 , # Hz
                  'DipoleMoment': 5.182  # e a0
                  }
    a87['D2'] = { 'wl'  : 780.241209686  , # nm
                  'f'   : 384.2304844685 , # THz 
                  'E'   : 1.589049462    , # eV 
                  'link': ['S1/2','P3/2'] ,
                  #'S'   : { 1:{ 0: 1/9, 1: 5/18 , 2: 5/18, 3:0 } , 2: {  0: 0, 1: 1/18 , 2: 5/18, 3:7/9 } }, # Sacado del siddons
                  'DecayRate': 38.117e6 ,# Hz
                  'DipoleMoment': 5.177  # e a0
                  }


    # Para calcular los pesos relativos de las transiciones se utilizó 
    # la ecuación 4 del paper de Siddons:
    
    for isotopo in [a85,a87]:
        # Iteramos sobre las líneas
        for D in [1,2]:
            # Iteramos sobre las polarizaciones ... S+/- son ciruclares, S es lineal
            for q,S in zip([-1,0,1],['S-','S','S+']):
                II  = isotopo['I']
                Je  = eval( isotopo[f'D{D}']['link'][1][1:] )
                Le  = 0 if 'S'==isotopo[f'D{D}']['link'][1][0] else 1
                Jg  = eval( isotopo[f'D{D}']['link'][0][1:] )
                Lg  = 0 if 'S'==isotopo[f'D{D}']['link'][0][0] else 1
                
                # Itero sobre los posible estados de Fg y Fe
                Fgs = [ int(a) for a in arange( abs(Jg-II) , abs(Jg+II+1) ) ]
                Fes = [ int(a) for a in arange( abs(Je-II) , abs(Je+II+1) ) ]
                
                isotopo[f'D{D}'][S] = {}
                for Fg in Fgs:
                    isotopo[f'D{D}'][S][Fg] = {}
                    for Fe in Fes:
                        valor = Cf2(Fg=Fg,Jg=Jg,Lg=Lg,Fe=Fe,Je=Je,Le=Le,q=q,II=II,S=1/2)
                        isotopo[f'D{D}'][S][Fg][Fe] = valor
                        print('.')


    with open('Rb.json', 'w') as fp:
        json.dump(atom, fp, sort_keys=True, indent=4, separators=(',', ': '), ensure_ascii=False)




#%% Simulación de espectros

#    La clase ATOM permite cargar los valores de un archivo .json y 
#    producir las simulaciones correspondientes

import json

from matplotlib.lines import Line2D
from scipy.special import wofz
from scipy.integrate import ode


# Algunas constantes útiles para el cálculo

a0   = 0.52917720859e-10 # m      Radio de Bohr
hbar = 1.054571628e-34   # J s    h/2 pi
cef  = 1/137.035999046   # Constante de estructura fina
KeV  = 8.6173324e-5      # eV/K    Constante de Boltzman
Kmks = 1.3806488e-23     # J/K     Constante de Boltzman
c=299792458              # m/s  Velocidad de la luz


# Clase auxiliar para convertir json en estrcutura de objeto
class data_struct(object):
    """Comment removed"""
    def __init__(self, data):
        for name, value in data.items():
            if not type(name) == str:
                name='i'+str(name)
            if '/' in name:
                name = name.replace('/','_')
            setattr(self, name, self._wrap(value))

    def _wrap(self, value):
        if isinstance(value, (tuple, list, set, frozenset)): 
            return type(value)([self._wrap(v) for v in value])
        else:
            return data_struct(value) if isinstance(value, dict) else value

class ATOM():
    """
    Clase para modelar espectros a absorción de un átomo con sus transiciones
    """
    def __init__(self,filename, DD=1 ):
        self.load_json(filename)
        self.DD = 'D1'            # Linea de trabajo
        self.set_D(DD)
        self.isotopos = [85,87]   # Isótopos
        #self.f0 = self.atom['isotopes'][  list(self.atom['isotopes'].keys())[0]  ][self.DD]['f']*1e12
        self.T=25+273      # Grados K
        self.Num = 20000
        self.L  = 0.075    # Longitud de la celda en metros
        self.presure_equation = 0   # Choos preasure equation
        self.calc_choose      = []
        self.spread           = 10000
        #self.name = self.atom['name']
        
        self._x_prim  = logspace(-10,10,100000)
        self._y_prim  = self._primitiva(self._x_prim)
        self._x_prim  = self._x_prim[self._y_prim.argsort()]
        self._y_prim  = self._y_prim[self._y_prim.argsort()]
    
    def set_D(self,DD):
        """
        Seleccionar línea D1 (S1/2 --> P1/2) o D2 (S1/2 --> P3/2)
        DD=1 o DD=2
        """
        if not DD in [1,2]:
            raise ValueError('DD should be 1 or 2')
        
        self.DD = f'D{DD}'
        self.f0 = self.atom['isotopes'][  list(self.atom['isotopes'].keys())[0]  ][self.DD]['f']*1e12
        self.l0 = c/self.f0
    
    def load_json(self,filename):
        """
        Cargar datos del json
        """
        with open(filename, 'r') as f:
            self.atom = json.load(f ,  object_hook=lambda d: {int(k) if k.lstrip('-').isdigit() else k: v for k, v in d.items()} )
            #self.__dict__.update(atom)
            for name, value in self.atom.items():
                if not type(name) == str:
                    name='i'+str(name)
                setattr(self, name, self._wrap(value))
    
    def _wrap(self, value):
        if isinstance(value, (tuple, list, set, frozenset)): 
            return type(value)([self._wrap(v) for v in value])
        else:
            return data_struct(value) if isinstance(value, dict) else value
        
    def __repr__(self):
        txt = self.symbol #+ '  isotopes: ' + ','.join( [ str(y) for y in self.isotopes.keys()] )
        return txt
    
    def calc_transitions(self):
        """
        Calcular transiciones desde los datos
        """
        
        #K_i    = []
        nu_i    = []
        Isat_i  = []
        sig_i   = []
        iso_i   = []
        mass_i  = []
        gamma_i = []
        name    = []
        II_i    = []
        SFFp_i  = []
        # Itero para cada isótopo
        for isot in self.isotopos:
            estado = self.atom['isotopes'][isot][self.DD]['link']
            DM     = self.atom['isotopes'][isot][self.DD]['DipoleMoment']
            gamma  = self.atom['isotopes'][isot][self.DD]['DecayRate']
            Gamma  = gamma/2/pi
            II     = self.atom['isotopes'][isot]['I']
            
            # Itero sobre los estados hiperfinos del fundamental
            for Fi,F in self.atom['isotopes'][isot]['levels'][estado[0]].items():
                # Itero sobre los estaos hiperfinos exitados
                for Fip, Fp in self.atom['isotopes'][isot]['levels'][estado[1]].items():
                    # Reglas de selección
                    if abs(Fip-Fi)<2: 
                        # Leo los coefieficientes de line strenght de esa transicion
                        SFFp     = self.atom['isotopes'][isot][self.DD]['S'][Fi][Fip]
                        
                        # Leo el omega de la transicion en Hz
                        nu_base = self.atom['isotopes'][isot][self.DD]['f']*1e12
                        
                        # frecuencia óptica de transicion
                        nu        = (Fp-F)*1e6 + nu_base 
                        
                        
                        gamma_eff = gamma  
                        gamma_eff *= self.atom['isotopes'][isot][self.DD]['S'][Fi][Fip]
                        gamma_eff /= sum( [ self.atom['isotopes'][isot][self.DD]['S'][fi][Fip]  for fi in arange(Fip-1,Fip+2) if fi in  self.atom['isotopes'][isot][self.DD]['S'].keys()]  )
                        
                        # Calculo I de saturacion
                        Isat_tmp =  gamma_eff**2*hbar/( 16*pi*cef * (DM**2* SFFp  * a0**2)  )
                        
                        # Incluyo las correcciones por degeneración de múltiples proyecciones
                        Isat_tmp *= (2*Fi+1) / (2*Fip+1)
                        
                        # Es la expresión que está en Steck, y que surge de corregir la saturación para 3 niveles
                        # http://steck.us/alkalidata
                        
                        sigma     = hbar*nu*2*pi*gamma_eff/2/Isat_tmp /9
                                                
                        # Caluclo valores de cada transicion y los guardo en cada array
                        nu_i    += [ nu                                  ]
                        Isat_i  += [ Isat_tmp                            ]
                        sig_i   += [ sigma                               ]
                        iso_i   += [ isot                                ]
                        mass_i  += [ self.atom['isotopes'][isot]['mass'] ]
                        gamma_i += [ gamma_eff                           ]
                        II_i    += [ II                                  ]
                        SFFp_i  += [ SFFp                                ]
                        name    += [ f"{isot:2d}Rb  {estado[0]:5s} F={Fi:1d} --> {estado[1]:5s} F'={Fip:1d}" ]
                        
        
        self.nu    = array( nu_i    )  # [Hz]
        self.Isat  = array( Isat_i  )  # [W / m²]
        self.sig   = array( sig_i   )  #  [mm²]    cross section area
        self.iso   = array( iso_i   )  # Isotopo
        self.mass  = array( mass_i  )  # Masa en Kg
        self.gamma = array( gamma_i )  # decaimiento en Hz (es 2*pi*Df)
        self.II    = array( II_i    )  # Spin del nucleo
        self.SFFp  = array( SFFp_i  )  # Factores de peso
        self.names = name              # Para identificar las transiciones
        self.calc_choose = [True]*len(self.names)
    
    
    def f(self,x, I):
        """
        Función para definir el sistema de ecuaciones diferenciales a integrar por ODE
        """
        
        rta = zeros(self.Num)
        names = array(self.names)
        for gamma,nui,sig,nRb,sigma_nu,isat,iso,nam in zip(self.gamma[self.calc_choose],self.nu[self.calc_choose],self.sig[self.calc_choose],self.nRb[self.calc_choose],self.sigma_nu[self.calc_choose],self.Isat[self.calc_choose],self.iso[self.calc_choose],names[self.calc_choose]):
            
            rta += 1/sqrt(1+I/isat)   * \
                   sig * nRb * \
                   gamma /sigma_nu * 1/(4*sqrt(2*pi))  * \
                   real(wofz(
                           ( (self.ff-nui) + 1j*(gamma/2*sqrt(1+I/isat)/2/pi))/(sigma_nu*sqrt(2))  
                           )) 
        return - I * rta 
    
    def calc_none(self):
        self.calc_choose = [False]*len(self.names)
    def calc_all(self):
        self.calc_choose = [True]*len(self.names)
    def calc_add(self,idd):
        if type(idd)==str:
            self.calc_choose = (( array(self.names) == idd ) | array(self.calc_choose)).tolist()
        elif type(idd)==int:
            self.calc_choose[idd] = True
    
    def plot_transitions(self,ax=[],decorate=False,lw=4):
        if ax == []:
            fig, ax = plt.subplots(1,1, figsize=(8,5),  constrained_layout=True )
            decorate=True
            
        for nu,s,isot in zip(self.nu,self.gamma,self.iso):
            #print(nu-self.f0)
            ax.plot( [ (nu-self.f0)/1e6 ]*2, [0,s], linewidth=lw, color='C'+str((isot-85)//2))
        
        if decorate:
            ax.grid(b=True,linestyle='--',color='lightgray')
            ax.set_xlabel(r'$\Delta \nu$ [MHz]')
            ax.set_ylabel(r'$\sigma_0$ [$\mu$m${}^2$]')
        
            custom_lines = [Line2D([0], [0], color='C0', lw=4),
                            Line2D([0], [0], color='C1', lw=4)]
            ax.legend(custom_lines, ['${}^{85}$Rb','${}^{87}$Rb'], loc='best')
        return ax


    def presure(self,T=0):
        """
        Función para el cáluculo de la presión a partir de la temperatura.
        Hay dos fuentes muy similares que se pueden elegir:
            con self.presure_equation=0 se usa la de Siddons (https://doi.org/10.1088/0953-4075/41/15/155004)
            con self.presure_equation=0 se usa la de Steck (http://steck.us/alkalidata)
        """
        
        if T==0:
            T=self.T
        Tcrit = 39.31 + 273.15
        if self.presure_equation>0:
            if T<Tcrit:
                return 133.323 * 10**( -94.04826 - 1961.258/T - 0.03771687*T + 42.57526*log10(T) )
            else:
                return 133.323 * 10**(  15.88253 - 4529.635/T + 0.00058663*T -  2.99138*log10(T) )
        else:
            return 133.322 * 10** ( (2.881+4.857-4215/T) if T<Tcrit else (2.881+4.312-4040/T) )
    
    
    def jac(self,x, I):
        return sum(-I/self.Isat/2 *  self.K/sqrt(1+I/self.Isat)**3 + self.K/sqrt(1+I/self.Isat) )
    
    
    def _primitiva(self,x):
        """
        Funcion auxiliar para aproximar integral de I
        """
        return -( 2*sqrt(1+x)+log(sqrt(1+x)-1)-log(sqrt(1+x)+1) )
    
    def _primitiva_inv(self,y):
        """
        Funcion inversa auxiliar para aproximar integral de I
        """
        return interp( y, self._y_prim , self._x_prim )
    
    
    
    def calc_transmision2(self, T=None, I=0):
        """
        Calcular transmisión usando aproximación integral
        """
        if T==None:
            T = self.T
        T            += 273.15 
        RbPresure     = self.presure( T )
        
        nRb           = (RbPresure/(Kmks*T))  # Num/m3
        sigma_nu      = self.nu/c * sqrt(Kmks*T/self.mass)
        nRbv          = array( [ nRb*0.7217 if y==85 else nRb*0.2783 for y in self.iso ] )
        self.nRb      = array( nRbv )
        self.sigma_nu = array( sigma_nu )
        names         = array(self.names)
        
        
        self.ff       = self.f0 + linspace(-self.spread,self.spread,self.Num)*1e6 
        if isinstance(I, (int, long, float, complex,np.float)):
            if I==0:
                I=float(mean(self.Isat).astype(float))
            I = array([I]*self.Num).astype(float)
        
        kappa        = self.ff *0
        Is           = self.ff *0
        for gamma,nui,sig,nRb,sigma_nu,isat,iso,nam in zip(self.gamma[self.calc_choose],self.nu[self.calc_choose],self.sig[self.calc_choose],self.nRb[self.calc_choose],self.sigma_nu[self.calc_choose],self.Isat[self.calc_choose],self.iso[self.calc_choose],names[self.calc_choose]):
            
            kfactor  =     1/(4*sqrt(2*pi))/sigma_nu * \
                            gamma * sig   *nRb       * \
                            exp( -(self.ff- nui )**2 / (2*sigma_nu**2) )
            kappa    += kfactor
            Is       += kfactor*isat
        
        #Is = mean(self.Isat[self.calc_choose])
        # se usa como I de saturación un promedio pesado por el Kappa
        # así se pareec más a los Is involucrados en la transicion dominante de cada partes
        # del espectro
        Is /= kappa
        
        self.Iout = self._primitiva_inv( +kappa*self.L + self._primitiva(I/Is) ) *Is
        
        self.kappa = kappa 
        self.yI   = array(I)
        self.yS   = array(self.Iout) / self.yI
        self.yT   = array(self.Iout)
        self.xf   = array(self.ff - self.f0).flatten()/1e6
    
    
    
    def calc_transmision_sat(self, T=None, I=0 , Ip=0):
        """
        Calcular transmisión saturada
        """
        if T==None:
            T = self.T
        T            += 273.15 
        RbPresure     = self.presure( T )
        
        nRb           = (RbPresure/(Kmks*T))  # Num/m3
        sigma_nu      = self.nu/c * sqrt(Kmks*T/self.mass)
        nRbv          = array( [ nRb*0.7217 if y==85 else nRb*0.2783 for y in self.iso ] )
        self.nRb      = array( nRbv )
        self.sigma_nu = array( sigma_nu )
        names         = array(self.names)
        
        
        self.ff       = self.f0 + linspace(-self.spread,self.spread,self.Num)*1e6 
        if isinstance(I, (int, long, float, complex,np.float)):
            if I==0:
                I=float(mean(self.Isat).astype(float))
            I = array([I]*self.Num).astype(float)
        if isinstance(Ip, (int, long, float, complex,np.float)):
            if Ip==0:
                I=float(mean(self.Isat).astype(float))*30
            Ip = array([Ip]*self.Num).astype(float)
        
        kappa        = self.ff *0
        Is           = self.ff *0
        Is_norm      = self.ff *0
        for gamma,nui,sig,nRb,sigma_nu,isat,iso,nam in zip(self.gamma[self.calc_choose],self.nu[self.calc_choose],self.sig[self.calc_choose],self.nRb[self.calc_choose],self.sigma_nu[self.calc_choose],self.Isat[self.calc_choose],self.iso[self.calc_choose],names[self.calc_choose]):
            kfactor  =     1/(4*sqrt(2*pi))/sigma_nu * \
                            gamma * sig   *nRb       * \
                            exp( -(self.ff- nui )**2 / (2*sigma_nu**2) )
            
            Is_norm  += kfactor
            Is       += kfactor*isat
            
            for tag in [ aa for aa in self.names if aa[:-1]==nam[:-1] ]:
                jj  = self.names.index(tag)
                LLL = 1/(1+4*(  self.ff-(2*self.nu[jj]-self.ff*self.nu[jj]/nui)   )**2   /(self.gamma[jj]/2/pi * sqrt(1+Ip/self.Isat[jj])   )**2 )
                
                kfactor  *= 1/(  1+ I/self.Isat[jj] * LLL )
            kappa    += kfactor
        
        #Is = mean(self.Isat[self.calc_choose])
        # se usa como I de saturación un promedio pesado por el Kappa
        # así se pareec más a los Is involucrados en la transicion dominante de cada partes
        # del espectro
        Is /= Is_norm
        
        self.Iout = self._primitiva_inv( +kappa*self.L + self._primitiva(I/Is) ) *Is
        
        self.kappa = kappa 
        self.yI   = array(I)
        self.yS   = array(self.Iout) / self.yI
        self.yT   = array(self.Iout)
        self.xf   = array(self.ff - self.f0).flatten()/1e6
    
    def calc_transmision(self, T=None, Tb=None , Tp=None , I=0):
        """
        Calcular transmisión usando ODE
        """
        if T==None:
            if Tb==None:
                Tb = self.T
            else:
                Tb += 273.15
                
            if Tp==None:
                Tp = self.T
            else:
                Tp += 273.15
        else:
            Tb = T + 273.15
            Tp = T + 273.15
        alpha = self.gamma / self.Isat
        
        RbPresure     = self.presure( Tp )
        nRb           = (RbPresure/(Kmks*Tp))  # Num/m3
        sigma_nu      = self.nu/c * sqrt(Kmks*Tb/self.mass)
        nRbv          = array( [ nRb*0.7217 if y==85 else nRb*0.2783 for y in self.iso ] )
        self.nRb      = array( nRbv )
        self.sigma_nu = array( sigma_nu )
        
        
        self.ff = self.f0 + linspace(-self.spread,self.spread,self.Num)*1e6 
        self.r  = ode(self.f).set_integrator('vode', method='bdf')
        self.r.set_integrator('vode')
        
        
        if isinstance(I, (int, long, float, complex,np.float)):
            if I==0:
                I=float(mean(self.Isat).astype(float))
            I = array([I]*self.Num).astype(float)

        self.r.set_initial_value( I )

        self.Iout = self.r.integrate( self.L )
        
        self.yI   = array(I)
        self.yS   = array(self.Iout) / self.yI
        self.yT   = array(self.Iout)
        self.xf   = array(self.ff - self.f0).flatten()/1e6
        
        
    def plot_transmision(self, ax=[], decorar=False ,**kwargs):
        if ax == []:
            fig,ax = plt.subplots(1,1, figsize=(8,5),  constrained_layout=True )
            decorar=True
        
        
        ax.plot( self.xf, self.yS , **kwargs) 
        
        if decorar:
            ax.grid(b=True,linestyle='--',color='lightgray')
            ax.set_xlabel(r'$\Delta \nu$ [MHz]')
            ax.set_ylabel(r'Transmitancia')
            ax.legend(loc='best')
        return ax
        
    def plot_temp_array(self, Tc=[ 0, 20, 50, 100, 300 ] , I=0):
        if type(Tc)==float or type(Tc)==int:
            Tc = [Tc]
        
        self.calc_transitions()
        gs_kw   = dict(width_ratios=[1], height_ratios=[1,2])
        fig,ax = plt.subplots(1,2, figsize=(8,5), gridspec_kw=gs_kw ,  constrained_layout=True )
        self.plot_transitions(ax=ax)
        ax[0].set_title('I='+str(I))
        
        Tv = array( Tc )+273
        
        for i,T in enumerate(Tv):
            self.T = T
            self.calc_transmision( I=I )
            self.plot_transmision(ax=ax[1], label=str(T-273)+' °C')
        
        ax[1].legend(loc='best')
        return ax


rb = ATOM('Rb.json')

rb.calc_transitions()



#%% Barrido de intensidad de haz en espectro de absorción

if __name__ == "__main__":

    fig,axx   = plt.subplots(1,2,figsize=(9,5), constrained_layout=True , sharey=True )
    barrido_I = [1.6,16,160,1600]
    T         = 25
    
    ax = axx[0]
    rb.set_D(2)
    rb.calc_transitions()
    for I in barrido_I:
        rb.calc_transmision(I=I,T=T)
        rb.plot_transmision(ax=ax , label=f'${I/10}$ mW/cm²')
    
        rb.calc_transmision2(I=I,T=T)
        rb.plot_transmision(ax=ax , ls=':', color=ax.lines[-1].get_color())
    ax.set_xlim(-5000,6000)
    ax.set_title(f'Línea {rb.DD}, temperatura ${T}^\circ$C')
    
    T         = 35
    ax = axx[1]
    rb.set_D(1)
    rb.calc_transitions()    
    
    for I in barrido_I:
        rb.calc_transmision(I=I,T=T)
        rb.plot_transmision(ax=ax )
    
        rb.calc_transmision2(I=I,T=T)
        rb.plot_transmision(ax=ax  , ls=':', color=ax.lines[-1].get_color())
    ax.set_xlim(-5000,6000)
    ax.set_title(f'Línea {rb.DD}, temperatura ${T}^\circ$C')
    
    h1, l1 = axx[0].get_legend_handles_labels()
    fig.legend(h1, l1, loc='center')
    
    for ax in axx:
        ax.set_xlabel(r"$\Delta \nu$ [MHz]")
        ax.set_ylim(0,1.05)
    axx[0].set_ylabel(r"Transmitancia")
    fig.suptitle(f"Espectro de absorción Rb para diferentes intensidades del láser")

    fig.savefig('modelo_Rb_01.png')




#%% Barrido de temperaturas en espectro de absorción

if __name__ == "__main__":

    fig,axx = plt.subplots(1,2,figsize=(9,5), constrained_layout=True , sharey=True )
    barrido_temperatura = [0,10,20,30,40]
    I = 16*0.1
    
    ax = axx[0]
    rb.set_D(2)
    rb.calc_transitions()
    for T in barrido_temperatura:
        rb.calc_transmision(I=I,T=T)
        rb.plot_transmision(ax=ax , label=f'${T}^\\circ$C')
    
        rb.calc_transmision2(I=I,T=T)
        rb.plot_transmision(ax=ax , ls=':', color=ax.lines[-1].get_color())
    ax.set_xlim(-5000,6000)
    ax.set_title(f'Línea {rb.DD}')
    
    ax = axx[1]
    rb.set_D(1)
    rb.calc_transitions()    
    
    for T in barrido_temperatura:
        rb.calc_transmision(I=I,T=T)
        rb.plot_transmision(ax=ax )
    
        rb.calc_transmision2(I=I,T=T)
        rb.plot_transmision(ax=ax  , ls=':', color=ax.lines[-1].get_color())
    ax.set_xlim(-5000,6000)
    ax.set_title(f'Línea {rb.DD}')
    
    h1, l1 = axx[0].get_legend_handles_labels()
    fig.legend(h1, l1, loc='center')
    
    for ax in axx:
        ax.set_xlabel(r"$\Delta \nu$ [MHz]")
        ax.set_ylim(0,1.05)
    axx[0].set_ylabel(r"Transmitancia")
    fig.suptitle(f"Espectro de absorción Rb para diferentes temperaturas")
    
    fig.savefig('modelo_Rb_02.png')


#%% Espectro de absorción saturada



if __name__ == "__main__":
    I           = 16*0.1    # W/m²
    T           = 20        # Temperatura en grados Centigrados
    rb.L        = 0.075     # largo de la celda: 7.5 cm
    I_pump      = 16*10
    rb.set_D(2)             # Línea D2
    rb.presure_equation=0   # Ecuación para definir presión
    
    rb.calc_transitions()
    fig,ax = plt.subplots(1,1,figsize=(9,5), constrained_layout=True , sharex=True )
    
    rb.calc_transmision_sat(I=16*0.1,T=T, Ip=I_pump)
    rb.plot_transmision(ax=ax , color='C0')
    ax.set_xlim(-4000,5100)
    
    ax.set_xlabel(r"$\Delta \nu$ [MHz]")
    ax.set_ylabel(r"Transmitancia")
    ax.grid(b=True,linestyle= ':',color='lightgray')
    ax.set_title(f"Espectro de absorción saturada Rb, línea {rb.DD} (T=${T}^\circ$C , Ipump={I_pump/10} mW/cm² )")

    fig.savefig('modelo_Rb_03.png')
    
    
