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
<center>
<strong>
<a href="cap09/DriverPZT">cap09/DriverPZT</a>
</strong>
</center>
-------
