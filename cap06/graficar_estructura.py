# -*- coding: utf-8 -*-
"""
Graficos de transiciones atómicas para el Rubidio
"""

from numpy import *
import matplotlib.pyplot as plt


#%% RUBIDIO 87 SOLAMENTE

# Cargamos los datos de la estructura atómica
# Obtenidos de http://steck.us/alkalidata

rb={}

rb[87]={}

rb[87]['S'] = {}
rb[87]['S']['F'] = {}
rb[87]['S']['F'][1] = { 'f' : -4271.676631815181 , # MHz 
                        'gF': -0.7 } # MHz/GHz
rb[87]['S']['F'][2] = { 'f' : +2563.005979089109 ,# MHz
                        'gF': +0.7 } # MHz/GHz

rb[87]['S']['f'] = 0 # MHz == 377 THz
rb[87]['S']['l'] = 0 # nm

rb[87]['P1/2'] = {}
rb[87]['P1/2']['F'] = {}
rb[87]['P1/2']['F'][1] = { 'f' : -509.05 , # MHz 
                           'gF': -0.23 } # MHz/GHz
rb[87]['P1/2']['F'][2] = { 'f' : +305.43 , # MHz 
                           'gF': +0.23 } # MHz/GHz
rb[87]['P1/2']['f'] = 377107463.380 # MHz == 377 THz
rb[87]['P1/2']['l'] = 794.978851156 # nm


rb[87]['P3/2'] = {}
rb[87]['P3/2']['F'] = {}
rb[87]['P3/2']['F'][0] = { 'f' : -302.0738 , # MHz 
                           'gF': 0.93 } # MHz/GHz
rb[87]['P3/2']['F'][1] = { 'f' : -229.8518 , # MHz 
                           'gF': 0.93 } # MHz/GHz
rb[87]['P3/2']['F'][2] = { 'f' : -72.9112 , # MHz 
                           'gF': 0.93 } # MHz/GHz
rb[87]['P3/2']['F'][3] = { 'f' : +193.7407 , # MHz 
                           'gF': 0.93 } # MHz/GHz
rb[87]['P3/2']['f'] = 384230484.4685 # MHz == 377 THz
rb[87]['P3/2']['l'] = 780.241209686 # nm




# Parche para visaulizar mejor
rb[87]['P1/2']['f'] = 377107463.380/2 # MHz == 377 THz

rb[87]['P3/2']['f'] = 384230484.4685 # MHz == 377 THz
rb[87]['S']['f'] = 0 # MHz == 377 THz



Bz = 0 # en Gauss

escala = {'S': 0.1 , 'P1/2':1, 'P3/2':1,}

escala_optica = 1/300000*2

fig,ax = plt.subplots(1,1, figsize=(12,7))

pls = {}

# Recorremos todos los niveles y graficamos lines y proyecciones

for level,item in rb[87].items():
    ax.plot([0,4.5] , ones(2)*item['f']*escala_optica , color='lightgray', linewidth=0.5)
    ax.plot([0,0.8] , ones(2)*item['f']*escala_optica , color='black', linewidth=2)
    
    ax.text(mean([0,0.8]), item['f']*escala_optica , level ,  horizontalalignment='center',
                                                              verticalalignment='bottom',
                                                              fontsize=12,
                                                              color='red')
    a_start=[0.8,item['f']*escala_optica]
    
    delta_f = []
    for Fn,F in item['F'].items():
        pls[Fn] = {}
        coord_y = item['f']*escala_optica+F['f']*escala[level]
        delta_f.append([coord_y,F['f']])
        ax.plot([1,1.8] , ones(2)*coord_y , color='black', linewidth=2)
        a_stop = [1,coord_y]
        
        ax.text(mean([1,1.8]), coord_y , 'F={:d}'.format(Fn),    horizontalalignment='center',
                                                                 verticalalignment='bottom',
                                                                 fontsize=12,
                                                                 color='red')
        
        arr=array([a_start,a_stop])
        
        ax.plot(arr[:,0] , arr[:,1] , '--' , color='gray')
        
        for mf in arange(-Fn,Fn+1):
            pls[Fn][mf], = ax.plot([2.4+1+mf/2,2.4+1+0.4+mf/2] , ones(2)*coord_y + mf*F['gF']*Bz , color='blue', linewidth=2)
    
    delta_f = array(delta_f)
    
    for ii,df in enumerate(delta_f):
        if ii==0:
            continue
        ax.annotate("", xy=(1.1, df[0]), xytext=(1.1, delta_f[ii-1][0]),
                          arrowprops=dict(arrowstyle="<->", connectionstyle="arc3") )
        ax.text(1.15, mean([delta_f[ii-1][0],df[0]]) , '{:5.2f} MHz'.format(-delta_f[ii-1][1]+df[1]) )
        print(ii)
        
    
xtks = []
xlbl = []
for mf in arange(-3,4):
    xtks.append( mean( [2.4+1+mf/2,2.4+1+0.4+mf/2] ))
    xlbl.append( '{:+02d}'.format(mf) )

ax.set_xticks(      xtks )
ax.set_xticklabels( xlbl )

ax.set_yticks([])
#ax.axis('off')

ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)





#%%

from matplotlib.widgets import Slider, Button, RadioButtons



gs_kw   = dict(width_ratios=[1], height_ratios=[10,1])
fig,axx = plt.subplots(2,1,figsize=(12,7), gridspec_kw=gs_kw )



Bz = 50 # en Gauss
escala = {'S': 0.1 , 'P1/2':1, 'P3/2':1,}
escala_optica = 1/300000*2
pls = {}

ax=axx[0]

for level,item in rb[87].items():
    pls[level] = {}
    ax.plot([0,4.5] , ones(2)*item['f']*escala_optica , color='lightgray', linewidth=0.5)
    ax.plot([0,0.8] , ones(2)*item['f']*escala_optica , color='black', linewidth=2)
    
    ax.text(mean([0,0.8]), item['f']*escala_optica , level ,  horizontalalignment='center',
                                                              verticalalignment='bottom',
                                                              fontsize=12,
                                                              color='red')
    a_start=[0.8,item['f']*escala_optica]
    
    delta_f = []
    for Fn,F in item['F'].items():
        pls[level][Fn] = {}
        coord_y = item['f']*escala_optica+F['f']*escala[level]
        delta_f.append([coord_y,F['f']])
        ax.plot([1,1.8] , ones(2)*coord_y , color='black', linewidth=2)
        a_stop = [1,coord_y]
        
        ax.text(mean([1,1.8]), coord_y , 'F={:d}'.format(Fn),    horizontalalignment='center',
                                                                 verticalalignment='bottom',
                                                                 fontsize=12,
                                                                 color='red')
        
        arr=array([a_start,a_stop])
        
        ax.plot(arr[:,0] , arr[:,1] , '--' , color='gray')
        
        for mf in arange(-Fn,Fn+1):
            pls[level][Fn][mf], = ax.plot([2.4+1+mf/2,2.4+1+0.4+mf/2] , ones(2)*coord_y + mf*F['gF']*Bz , color='blue', linewidth=2)
    
    delta_f = array(delta_f)
    
    for ii,df in enumerate(delta_f):
        if ii==0:
            continue
        #ax.arrow(1.4,df[0],0,-df[0]+delta_f[ii-1][0] , head_length=50, head_width=0.1,head_starts_at_zero=True )
        ax.annotate("", xy=(1.1, df[0]), xytext=(1.1, delta_f[ii-1][0]),
                          arrowprops=dict(arrowstyle="<->", connectionstyle="arc3") )
        ax.text(1.15, mean([delta_f[ii-1][0],df[0]]) , '{:5.2f} MHz'.format(-delta_f[ii-1][1]+df[1]) )
        print(ii)
        
    
xtks = []
xlbl = []
for mf in arange(-3,4):
    xtks.append( mean( [2.4+1+mf/2,2.4+1+0.4+mf/2] ))
    xlbl.append( '{:+02d}'.format(mf) )

ax.set_xticks(      xtks )
ax.set_xticklabels( xlbl )





sl = Slider(axx[1], 'Bz' , -100, 100 , valinit=0) 


def update(Bz):
    
    for level,lo in pls.items():
        for F,Fo in lo.items():
            coord_y =  rb[87][level]['f']*escala_optica+rb[87][level]['F'][F]['f']*escala[level]
            for mf,mfo in Fo.items():
                mfo.set_ydata( ones(2)* coord_y + mf*rb[87][level]['F'][F]['gF']*Bz  )
                #print(level,F,mf)
    #fig.canvas.draw_idle()


sl.on_changed(update)









#%% RUBIDIO 87 y 85


rb={}


rb[87]={}

rb[87]['S'] = {}
rb[87]['S']['F'] = {}
rb[87]['S']['F'][1] = { 'f' : -4271.676631815181 , # MHz 
                        'gF': -0.7 } # MHz/GHz
rb[87]['S']['F'][2] = { 'f' : +2563.005979089109 ,# MHz
                        'gF': +0.7 } # MHz/GHz

rb[87]['S']['f'] = 0 # MHz == 377 THz
rb[87]['S']['l'] = 0 # nm

rb[87]['P1/2'] = {}
rb[87]['P1/2']['F'] = {}
rb[87]['P1/2']['F'][1] = { 'f' : -509.05 , # MHz 
                           'gF': -0.23 } # MHz/GHz
rb[87]['P1/2']['F'][2] = { 'f' : +305.43 , # MHz 
                           'gF': +0.23 } # MHz/GHz
rb[87]['P1/2']['f'] = 377107463.380 # MHz == 377 THz
rb[87]['P1/2']['l'] = 794.978851156 # nm


rb[85]={}
rb[85]['S'] = {}
rb[85]['S']['F'] = {}
rb[85]['S']['F'][2] = { 'f' : -1770.843922 , # MHz 
                        'gF': -0.47 } # MHz/GHz
rb[85]['S']['F'][3] = { 'f' : +1264.8885163 , # MHz 
                        'gF': +0.47 } # MHz/GHz
rb[85]['S']['f'] = 0 # MHz == 377 THz
rb[85]['S']['l'] = 0 # nm

rb[85]['P1/2'] = {}
rb[85]['P1/2']['F'] = {}
rb[85]['P1/2']['F'][2] = { 'f' : -210.923 , # MHz 
                           'gF': -0.16 } # MHz/GHz
rb[85]['P1/2']['F'][3] = { 'f' : +150.659 , # MHz 
                           'gF': +0.16 } # MHz/GHz
rb[85]['P1/2']['f'] = 377107385.690 # MHz == 377 THz
rb[85]['P1/2']['l'] = 794.979014933 # nm


#rb[87]['D1'] = rb[87]['P1/2']


Bz = 0 # en Gauss

escala = {'S': 0.1 , 'P1/2':1, 'P3/2':1,}

escala_optica = 1/300000



polarizacion = +1


for i0 in [0,1]:
    for i1 in [0,1]:
        
        fig,axx = plt.subplots(1,2, figsize=(8,5))
        
        pls = {}
        
        mf_data = {}
        
        for isotopo,ax in zip( [87,85] , axx ):
            ax.set_title('${}^{'+str(isotopo)+'}Rb$')
            mf_data[isotopo] = {}
            for level,item in rb[isotopo].items():
                ax.plot([0,4.5] , ones(2)*item['f']*escala_optica , color='lightgray', linewidth=0.5)
                ax.plot([0,0.8] , ones(2)*item['f']*escala_optica , color='black', linewidth=2)
                
                ax.text(mean([0,0.8]), item['f']*escala_optica , level ,  horizontalalignment='center',
                                                                          verticalalignment='bottom',
                                                                          fontsize=12,
                                                                          color='red')
                a_start=[0.8,item['f']*escala_optica]
                
                delta_f  = []
                mf_data[isotopo][level] = {}
                for Fn,F in item['F'].items():
                    mf_data[isotopo][level][Fn] = {}
                    pls[Fn] = {}
                    coord_y = item['f']*escala_optica+F['f']*escala[level]
                    delta_f.append([coord_y,F['f']])
                    ax.plot([1,1.8] , ones(2)*coord_y , color='black', linewidth=2)
                    a_stop = [1,coord_y]
                    
                    ax.text(mean([1,1.8]), coord_y , 'F={:d}'.format(Fn),    horizontalalignment='center',
                                                                             verticalalignment='bottom',
                                                                             fontsize=12,
                                                                             color='red')
                    
                    arr=array([a_start,a_stop])
                    
                    ax.plot(arr[:,0] , arr[:,1] , '--' , color='gray')
                    
                    # Cada MF
                    #delta_mf[Fn] = {}
                    for mf in arange(-Fn,Fn+1):
                        pls[Fn][mf], = ax.plot([2+1.5+mf/2,2+1.5+0.4+mf/2] , ones(2)*coord_y + mf*F['gF']*Bz , color='blue', linewidth=2)
                        mf_data[isotopo][level][Fn][mf] = ( mean([2+1.5+mf/2,2+1.5+0.4+mf/2]) , coord_y + mf*F['gF']*Bz )
                delta_f = array(delta_f)
                
                for ii,df in enumerate(delta_f):
                    if ii==0:
                        continue
                    #ax.arrow(1.4,df[0],0,-df[0]+delta_f[ii-1][0] , head_length=50, head_width=0.1,head_starts_at_zero=True )
                    ax.annotate("", xy=(1.1, df[0]), xytext=(1.1, delta_f[ii-1][0]),
                                      arrowprops=dict(arrowstyle="<->", connectionstyle="arc3") )
                    ax.text(1.15, mean([delta_f[ii-1][0],df[0]]) , '{:5.2f} MHz'.format(-delta_f[ii-1][1]+df[1]) )
                    print(ii)
                    
                
            xtks = []
            xlbl = []
            for mf in arange(-3,4):
                xtks.append( mean( [2+1.5+mf/2,2+1.5+0.4+mf/2] ))
                xlbl.append( '{:+02d}'.format(mf) )
            
            ax.set_xticks(      xtks )
            ax.set_xticklabels( xlbl )
            
            ax.set_yticks([])
            #ax.axis('off')
            
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            
            # Graficar Transiciones
            level0 = sort(list(mf_data[isotopo].keys()))[::-1][0]
            level1 = sort(list(mf_data[isotopo].keys()))[::-1][1]
            
            F0 = sort(list(mf_data[isotopo][level0].keys()))[i0]
            F1 = sort(list(mf_data[isotopo][level1].keys()))[i1]
            
            mf_ground = set(mf_data[isotopo][level0][F0].keys())
            
            for mf_start,start in mf_data[isotopo][level0][F0].items():
                for mf_stop,stop in mf_data[isotopo][level1][F1].items():
                    if mf_stop-mf_start == polarizacion:
                        print(mf_stop,mf_start)
                        ax.annotate("", xy=(stop[0], stop[1]), 
                                    xytext=(start[0], start[1]),
                                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color='C1') )
                        mf_ground.discard( mf_start )
            for mf in mf_ground:
                start = mf_data[isotopo][level0][F0][mf]
                ax.plot(start[0],start[1],'o', color='C1')
        
        axx[1].set_ylim(  axx[0].get_ylim() )
        axx[0].set_xlim(  axx[1].get_xlim() )
        
        
        fig.tight_layout()
        
        #plt.pause(0.1)
        #fig.savefig('20190825_'+str(i0)+str(i1)+'.png')
        







