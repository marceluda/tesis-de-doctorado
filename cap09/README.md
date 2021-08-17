# Capítulo 9: Espectroscopía de absorción de gas de Rb usando un microcontrolador Arduino

Se montó un sistema de actuadores para controlar un ECDL y realizar un experimento de espectroscopía de absorción saturada, como en este esquema:

<img src="esquema.png" alt="IMAGE ALT TEXT HERE"  border="10" />


Para ello se debió controlar:
  - Corriente del láser
  - Temperatura del láser
  - Posición de la red de difracción, emdiante una tensión en un PZT

Se desarrollaron circuitos controladores para cada una de estas variables físicas. También se desarrolló un sistema de instrumentación basado en Arduino.

## Controladores

### Controlador de PZT

Simplemente es un circuito de adaptación para el integrado [E-661.OE](hojas_de_datos/e-660_userpz45e223.pdf). Tiene un canal de control de off-set y otro de barrido.
El de off-set puede ser controlado por una entrada PWM  o mediante un potenciómetro multivuelta.
El de barrido por una entrada BNC o por otro PWM. Los canale se configuran mediante dos jumpres.

<img src="DriverPZT_brd.png" alt="IMAGE ALT TEXT HERE"  border="10" />
<img src="DriverPZT_sch.png" alt="IMAGE ALT TEXT HERE"  border="10" />


-------

<p align="center">
<strong>
<a href="DriverPZT">Driver PZT</a>
</strong>
</p>

-------


### Controlador de Corriente

Es un circuito de adaptación de tensiones para controlar el integrado [FL500](hojas_de_datos/fl500.pdf), que es una fuente de corriente de hasta 500 mA.
Tiene un control de off-set t un control de barrido, que incluye atenuación variable y la posibilidad de invertir la dirección de barrido respecto a la señal de control de entrada.


<img src="DriverCorriente_sch.png" alt="IMAGE ALT TEXT HERE"  border="10" />
<img src="DriverCorriente_brd1.png" alt="IMAGE ALT TEXT HERE"  border="10" width='250px' /> <img src="DriverCorriente_brd2.png" alt="IMAGE ALT TEXT HERE"  border="10" width='250px'  />

-------

<p align="center">
<strong>
<a href="DriverFL500">Driver Corriente</a>
</strong>
</p>

-------


### Controlador de Temperatura

Es un circuito de realimentación negativa con un controlador PID para estabilizar la temperatura del láser. Actúa sobre una celda Peltier y sensa un terminstor, que es comparado con una tensión de referencia.


<img src="DriverTEC_brd.png" alt="IMAGE ALT TEXT HERE"  border="10" />
<img src="DriverTEC_sch1.png" alt="IMAGE ALT TEXT HERE"  border="10" width='250px' /><img src="DriverTEC_sch2.png" alt="IMAGE ALT TEXT HERE"  border="10" width='250px'  />

-------

<p align="center">
<strong>
<a href="TEC_controller">Driver TEC</a>
</strong>
</p>

-------


## Controlador centra basado en Arduino

Es un *shield* para [Arduino MEGA 2560](https://store.arduino.cc/usa/mega-2560-r3) que permite incorporarle:

  * Generador de barridos traingulares
  * ADC con amplificación y offset

<img src="shield-control_sch.png" alt="IMAGE ALT TEXT HERE"  border="10" />
<img src="shield-control_brd1.png" alt="IMAGE ALT TEXT HERE"  border="10" width='250px' /><img src="shield-control_brd2.png" alt="IMAGE ALT TEXT HERE"  border="10" width='250px'  />



-------

<p align="center">
<strong>
<a href="shield-control">Arduino Shield-Control</a>
</strong>
</p>

-------
