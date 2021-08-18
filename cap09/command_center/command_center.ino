/*
  Controlamos el arduino con comandos
 */

#define NBUF 50


// Caracter End Of Transmision para controlar comunicacion
const char EOT=0x4;

// variables para debuggin
int debug_mode = 0;
char debug_buff[NBUF];
int debug_len=0;
unsigned long time0;
unsigned long time1;
unsigned long utime0;
unsigned long utime1;




// Varaibles para interaccion Serial
String buffer = "";   // buffer de texto que entra


char buff[NBUF];
String comando= "";
String param1="";    
String param2="";
int bufferLength = buffer.length();     // previous length of the String
char charBuff=' ';
int ind=0;
int esp1=0 ; int esp2=0 ;

// Varaibles para tratar los pines. Les pongo valor de inicio por si acaso, pero no hace falta
int pinNum= 12;
int pinNum2=12;
int pinVal=0;
String pinStr = "";
String valStr = "";

// Variables para guardar valores
byte pwm_val[]={0,0,0,0,0,0,0,0,0,0,0,0};
byte dig_val[]={0,0,0,0,0,0,0,0,0,0,0,0};

// Para levantar curvas
const int NN=1000;
unsigned int  curv0[NN];
//unsigned int  curv1[NN];
byte          curvD[NN];
unsigned int  curvT[NN];

// Variables de control
int binary_mode=0;
int Vref=5;

boolean delayMicro=false;
int Tmicro=100;
int Tmilli=1;

// Variables auxiliares
char tmp2[2];
char tmp4[4];



// Para toquetear los prescaler de los ADC
const unsigned char PS_16 = (1 << ADPS2);
const unsigned char PS_32 = (1 << ADPS2) | (1 << ADPS0);
const unsigned char PS_64 = (1 << ADPS2) | (1 << ADPS1);
const unsigned char PS_128 = (1 << ADPS2) | (1 << ADPS1) | (1 << ADPS0);


// Manejo de interrupciones *****************************
// Interrupt Service Routine (ISR)
volatile boolean trigFlag;
boolean trigger_mode=false;

void trigger()
{
  trigFlag=true;
}  // end of trigger




/*********************************************************************************************
                                          SETUP
**********************************************************************************************/


void setup() {     

  // Habilitamos el serial
  Serial.begin(57600);
  while (!Serial) {
    ; // wait for serial port to connect. Needed for Leonardo only
  }

 //Configuramos TIMERS para los PWMs ********************************************************
 
 /*
  timer 0 (controls pin 13, 4);
  timer 1 (controls pin 12, 11);
  timer 2 (controls pin 10, 9);
  timer 3 (controls pin 5, 3, 2);
  timer 4 (controls pin 8, 7, 6);
  timer 5 (controls pin 44, 45, 46);

  http://sobisource.com/?p=195
  
  TCCRnB  controla el timer 'n' con un num de 8-bit
  Hay que cambiar los 3 primeros bits:  CS02, CS01, CS00
  
  Para borrar:
  int myEraser = 7;  // 7 == 111
  TCCRnB &= ~myEraser;
  
  Para setear:
  int myPrescaler = 3;  // poner num entre [1 , 6]
  TCCR2B |= myPrescaler;
  
  prescaler = 1 ---> PWM frequency is 31000 Hz
  prescaler = 2 ---> PWM frequency is 4000 Hz
  prescaler = 3 ---> PWM frequency is 490 Hz (default value)
  prescaler = 4 ---> PWM frequency is 120 Hz
  prescaler = 5 ---> PWM frequency is 30 Hz
  prescaler = 6 ---> PWM frequency is <20 Hz
  
  Ojo, el CERO es el del sistema (de los pines 13 y 14) y tiene otros valores
  
  prescaler = 1 ---> PWM frequency is 62000 Hz
  prescaler = 2 ---> PWM frequency is 7800 Hz
  prescaler = 3 ---> PWM frequency is 980 Hz (default value)
  prescaler = 4 ---> PWM frequency is 250 Hz
  prescaler = 5 ---> PWM frequency is 60 Hz
  prescaler = 6 ---> PWM frequency is <20 Hz
  
  */
  
  int myEraser = 7;
  TCCR1B &= ~myEraser;  // pin 11 y 12
  TCCR2B &= ~myEraser;  // pin  9 y 10
  TCCR3B &= ~myEraser;  // pin  2, 3 y 5
  TCCR4B &= ~myEraser;  // pin  6, 7 y 8


  int myPrescaler = 1;
  TCCR1B |= myPrescaler;
  TCCR2B |= myPrescaler;
  TCCR3B |= myPrescaler;
  TCCR4B |= myPrescaler;



  delay(500);
  
  
  // Prescaler para los ADC
  ADCSRA &= ~PS_128;  // remove bits set by Arduino library
  // El valor por defecto es PS_128 ~ 120 us por punto relevado. 64 es la mitad y asi...
  // PS_16, PS_32, PS_64 or PS_128
  ADCSRA |= PS_32;  // ~ 30 us
  
  
  
  //Inicializamos variables del programa *****************************************************
  
  for(int i=0;i<NN;i++){
    curv0[i]=0;
    //curv1[i]=0;
    curvD[i]=0;
    curvT[i]=0;
  }
  
  // Inicializamos todos los pines de salida PWM
  for(int i=2; i<14;i++){
    pinMode(i, OUTPUT);
    if(debug_mode!=0){debug_len=sprintf(debug_buff,"Inicializando puerto %2d en OUTPUT\n",i);Serial.print(debug_buff);};
  }
  
  if(debug_mode!=0) Serial.println(" ");
  
  // Inicializamos todos los pines de salida DIG
  for(int i=30; i<40;i++){
    pinMode(i, OUTPUT);
    if(debug_mode!=0){debug_len=sprintf(debug_buff,"Inicializando puerto %2d en OUTPUT\n",i);Serial.print(debug_buff);};
  }
  
  if(debug_mode!=0) Serial.println(" ");
  
  // Inicializamos todos los pines de entrada DIG
  for(int i=24; i<30;i++){
    pinMode(i, INPUT);
    if(debug_mode!=0){debug_len=sprintf(debug_buff,"Inicializando puerto %2d en INPUT\n",i);Serial.print(debug_buff);};
  }
  
  if(debug_mode!=0) Serial.println(" ");
  
  // Inicializamos todos los pines de entrada ANALOGICA
  
  pinMode(A0, INPUT); // pin 54
  if(debug_mode!=0) Serial.println("Inicializando puerto A0 en INPUT");
  pinMode(A1, INPUT); // pin 55
  if(debug_mode!=0) Serial.println("Inicializando puerto A1 en INPUT");
  pinMode(A2, INPUT); // pin 56
  if(debug_mode!=0) Serial.println("Inicializando puerto A2 en INPUT");
  pinMode(A3, INPUT); // pin 57
  if(debug_mode!=0) Serial.println("Inicializando puerto A3 en INPUT");
  pinMode(A8, INPUT); // pin ??
  if(debug_mode!=0) Serial.println("Inicializando puerto A8 en INPUT");
  
  
  
  if(debug_mode!=0) Serial.println(' ');
  msg("ready\n");
  Serial.write(EOT);
  
  
  // Inicializamos los input trigger
  pinMode(19, INPUT);
  pinMode(20, INPUT);
  
  
  // LOLO
  analogWrite(10,150);
  binary_mode=1;
  
  //analogReference(EXTERNAL);
  //Vref=9;
} // fin SETUP


















/*********************************************************************************************
                                          MAIN LOOP
**********************************************************************************************/

void loop() {
  // ********************************* LEVANTAR CARACTERES ***********************************
  // Si recibo caracteres, los acumulo en charBuff
  while (Serial.available() > 0) {
    charBuff = Serial.read();
    // Si no es un "fin de linea" acumulo los caracteres en buffer
    if(charBuff!='\n') buffer += charBuff;
  }
  
  /*
    Los comandos se componen de la siguiente estructura:
    (comando)( )(param1)[ ][param2]
    donde () implica obligatorio, [] opcional y ( ) es un white_space
    
    
    debug modes
    0   --> nada de debug
    >=5 --> info irrelevante
    
    
    Esquema actual
    
    get -----> P(2-13)  ---> Trae ultimo valor seteado al pin Px
          |--> D(24-29) ---> Lee valor digital del pin Dx
          |--> D(30-39) ---> Trae ultimo valor seteado al pin Dx
          |--> A(0-4)   ---> Lee valor analogico del pin Ax (pin x+54)
    
    set -----> P(2-13) (0-255) ---> Setea el PWM Px en un valor 0-255
          |--> D(30-39) (0-1)  ---> Setea el pin digital Dx en un valor 0-1
          
    debug ---> (num)          ---> Setea el modo de debugging en el valor "num"
    
    trig ---> (off||up||down) ---> Habilita el trigger en el pin 19 para subida, bajada, o lo apaga
    
    binary --> (num)          ---> Si num>0 habilita el modo binario. Sino, lo deshabilita.
    
    vref ---> (num)   ---> Cambia la tension de referencia del ADC
                              |--> num=1  ---> Vref=1.1V
                              |--> num=2  ---> Vref=2.56V
                              |--> num=9  ---> Vref externo
                              |--> cualquier otro ---> Vref=Vcc~5V (DEFAULT)
          
    curv  ----> A(0-4)   ---> Levanta NN puntos del pin Ax
    
    curv2 ----> A(0-4) ----> D(19-20||30-39)   ---> Levanta NN puntos del pin Ax y otra de NN puntos del pin Dx
    
    En modo binario las respuestas son:
    
    --> 0 EOT                    ---> Comando recibido
    --> 1 EOT Val EOT            ---> Envio de un integer binario
    --> 2 EOT Num EOT Vals EOT   ---> Envio de Num integers binarios
    --> 9 EOT msg EOT            ---> Envio de un texto informativo
    
    
  */

  // ********************************* PROCESAR COMANDOS **************************************
  // Si recibo fin de linea, leo y ejecuto
  if (charBuff=='\n') {
    // Algunas funciones de formateo por si metemos esopacios de mas que no van:
    buffer.trim();
    buffer.replace("  "," ");
    buffer.replace("  "," ");
    buffer.replace("  "," ");
    
    
    
    
    if(buffer.length()>1){
      
      // donde esta el primer white_space del comando?
      esp1=buffer.indexOf(' ');
      
      // Comando es todo lo que viene antes del primer esapcio
      comando= buffer.substring(0,esp1);
      
      
      
      
      // Si el comando es SET *****************************************************************************
      if(comando.substring(0,3)=="set" || comando.substring(0,3)=="Set" || comando.substring(0,3)=="SET"){
        // donde esta el segundo white_space del comando?
        esp2=buffer.indexOf(' ',esp1+1);
        
        // param1 es lo que esta entre los dos espacios
        param1=buffer.substring(esp1+1,esp2);
        // param2 es lo que esta despues del segundo espacio
        param2=buffer.substring(esp2+1);
        
        // DEBUG
        if(debug_mode!=0){debug_len=sprintf(debug_buff,
                                            "comando: %s | param1: %s | param2: %s\n",
                                            comando.c_str(),
                                            param1.c_str(),
                                            param2.c_str()
                                            );Serial.print(debug_buff);};
                                            
        
        if(param1.substring(0,1)=="P" || param1.substring(0,1)=="D")
        {
          if(debug_mode>5) Serial.println("Vamos a cambiarle el valor a un pin");
          pinStr=param1.substring(1);
          pinStr.toCharArray(buff, NBUF);
          pinNum=atoi(buff);
          
          param2.toCharArray(buff, NBUF);
          pinVal=atoi(buff);
          
          if(param1.substring(0,1)=="P")
            if( pinNum>1 && pinNum<14 && pinVal>=0 && pinVal<256 ){
              analogWrite(pinNum,pinVal);
              pwm_val[pinNum-2]=(unsigned int)pinVal;
              if(debug_mode!=0){debug_len=sprintf(debug_buff,"PWM pinNum: %2d | pinVal: %3d \n",pinNum,pinVal);Serial.print(debug_buff);};
            }
          
          if(param1.substring(0,1)=="D")
            if( pinNum>=30 && pinNum<40 && pinVal>=0 && pinVal<=1 ){
              digitalWrite(pinNum, pinVal==0 ? LOW : HIGH);
              dig_val[pinNum-30]=(unsigned int)pinVal;
              if(debug_mode!=0){debug_len=sprintf(debug_buff,"DIG pinNum: %2d | pinVal: %3d \n",pinNum,pinVal);Serial.print(debug_buff);};
            }
            
            //Serial.print("ok");
            msg("0");
            msgEnd();
          
        }
      } // Fin del SET
      
      
      
      
      

      // Si el comando es GET *****************************************************************************
      else if(comando.substring(0,3)=="get" || comando.substring(0,3)=="Get" || comando.substring(0,3)=="GET"){
        // param1 es lo que esta despues del primer espacio
        param1=buffer.substring(esp1+1);
        
        // DEBUG
        if(debug_mode!=0){debug_len=sprintf(debug_buff,
                                            "comando: %s | param1: %s \n",
                                            comando.c_str(),
                                            param1.c_str()
                                            );Serial.print(debug_buff);};
                                            
        
        if(param1.substring(0,4)=="vref" || param1.substring(0,4)=="Vref" || param1.substring(0,4)=="VREF"){
            if(binary_mode>0){
              msg("1");
              msgEnd();
              Serial.write(lowByte(Vref));
              Serial.write(highByte(Vref));
            }else{
              Serial.print(Vref);
            }
            msgEnd();
        }
        if(param1.substring(0,5)=="delay" || param1.substring(0,5)=="Delay" || param1.substring(0,5)=="DELAY"){
            if(binary_mode>0){
              msg("9");
              msgEnd();
            }
            delayMicro ? Serial.print(Tmicro) : Serial.print(Tmilli);
            delayMicro ? Serial.print(" us")    : Serial.print(" ms");
            msgEnd();
        }
        
        else if(param1.substring(0,1)=="P" || param1.substring(0,1)=="D" || param1.substring(0,1)=="A")
        {
          if(debug_mode>5) Serial.println("Vamos a leer que le seteamos a un pin de OUTPUT");
          pinStr=param1.substring(1);
          pinStr.toCharArray(buff, NBUF);
          pinNum=atoi(buff);

          
          if(param1.substring(0,1)=="P")
            if( pinNum>1 && pinNum<14 ){
              pinVal=(int)pwm_val[pinNum-2];
              if(binary_mode>0){
                tmp2[0]=lowByte(pinVal);
                tmp2[1]=highByte(pinVal);
              }else{
                sprintf(debug_buff,"PWM P%d = %3d \n",pinNum,pinVal);Serial.print(debug_buff);
              }
            }
          
          if(param1.substring(0,1)=="D")
            if( pinNum>=30 && pinNum<40){
              pinVal=(int)dig_val[pinNum-30];
              if(binary_mode>0){
                tmp2[0]=lowByte(pinVal);
                tmp2[1]=highByte(pinVal);
              }else{
                sprintf(debug_buff,"DIG D%d = %3d \n",pinNum,pinVal);Serial.print(debug_buff);
              }
            }else if(pinNum>=24 && pinNum<30){
              pinVal=digitalRead(pinNum);
              if(binary_mode>0){
                tmp2[0]=lowByte(pinVal);
                tmp2[1]=highByte(pinVal);
              }else{
                sprintf(debug_buff,"DIG D%d = %3d \n",pinNum,pinVal);Serial.print(debug_buff);
              }
            }
          
          if(param1.substring(0,1)=="A")
            if( pinNum>=0 && pinNum<5){
              pinVal=analogRead(pinNum+54);
              if(binary_mode>0){
                tmp2[0]=lowByte(pinVal);
                tmp2[1]=highByte(pinVal);
              }else{
                sprintf(debug_buff,"ANALOG A%d = %3d \n",pinNum,pinVal);Serial.print(debug_buff);
              }
            }
            
            if(binary_mode>0){
              msg("1");
              msgEnd();
              Serial.write(tmp2[0]);
              Serial.write(tmp2[1]);
            }else{
              Serial.print(pinVal);
            }
            msgEnd();
            //Serial.println(pinVal);
            //msgEnd();
            
        }
      } // Fin del GET
      
      
      // Si el comando es ATE *****************************************************************************
      if(comando.substring(0,3)=="ate" || comando.substring(0,3)=="Ate" || comando.substring(0,3)=="ATE"){
        // param1 es lo que esta despues del 1er espacio
        param1=buffer.substring(esp1+1);
        
        // DEBUG
        if(debug_mode!=0){debug_len=sprintf(debug_buff,
                                            "comando: %s | param1: %s \n",
                                            comando.c_str(),
                                            param1.c_str()
                                            );Serial.print(debug_buff);};
                                            
        
        if(debug_mode>5) Serial.println("Vamos a cambiarle el valor al atenuador");
        param1.toCharArray(buff, NBUF);
        pinVal=atoi(buff);
        
        for(int i=0; i<4; i++){
          digitalWrite(30+2*i, bitRead(pinVal,i)==0 ? LOW : HIGH);
          dig_val[i*2]=(unsigned int) bitRead(pinVal,i);
        }
        
        //Serial.print("ok");
        msg("0");
        msgEnd();
          
      
      } // Fin del ATE
      
      
      
      // Si el comando es DEBUG ***************************************************************************
      else if(comando.substring(0,5)=="debug" || comando.substring(0,5)=="Debug" || comando.substring(0,5)=="DEBUG"){
        param1=buffer.substring(esp1+1);
        pinStr=param1.substring(0);
        pinStr.toCharArray(buff, NBUF);
        debug_mode=atoi(buff);
        if(binary_mode>0) {
          msg("9");
          msgEnd();
        }
        sprintf(debug_buff,"DEBUG nuevo valor debug_mode=%d\n",debug_mode);Serial.print(debug_buff);
        msgEnd();
      }
      
      
      
      
      // Si el comando es TRIG ***************************************************************************
      else if(comando.substring(0,4)=="trig" || comando.substring(0,4)=="Trig" || comando.substring(0,4)=="TRIG"){
        param1=buffer.substring(esp1+1);
        
        
        trigger_mode=false;
        
        if(param1.substring(0,2)=="up" || param1.substring(0,2)=="Up" || param1.substring(0,2)=="UP"){
            attachInterrupt (4, trigger, RISING);
            trigger_mode=true;
            //msg("trig UP\n");
        } // 4 es el pin19 y 5 es el pin18
        
        if(param1.substring(0,4)=="down" || param1.substring(0,4)=="Down" || param1.substring(0,4)=="DOWN"){
            attachInterrupt (4, trigger, FALLING); // 4 es el pin19 y 5 es el pin18
            trigger_mode=true;
            //msg("trig DOWN\n");
        }
        
        
        if(param1.substring(0,3)=="off" || param1.substring(0,3)=="Off" || param1.substring(0,3)=="OFF")
            detachInterrupt(4);
        
        if(binary_mode>0) {
          msg("9");
          msgEnd();
        }
        msg("Trigger en modo:"); Serial.print(trigger_mode);
        msgEnd();
      } // END TRIG
      
      

      
      
      
      // Si el comando es DELAY *****************************************************************************
      if(comando.substring(0,5)=="delay" || comando.substring(0,3)=="Delay" || comando.substring(0,3)=="DELAY"){
        // donde esta el segundo white_space del comando?
        esp2=buffer.indexOf(' ',esp1+1);
        
        if(esp2==-1){ // Si no hay segundo parametro
            param1=buffer.substring(esp1+1);
            param2="m";
        }else{
            // param1 es lo que esta entre los dos espacios
            param1=buffer.substring(esp1+1,esp2);
            // param2 es lo que esta despues del segundo espacio
            param2=buffer.substring(esp2+1);
            if(param2.substring(0,1)=="u" || param2.substring(0,1)=="U") 
                  param2="u"; 
                  else param2="m";
            
        }
        
        // DEBUG
        if(debug_mode!=0){debug_len=sprintf(debug_buff,
                                            "comando: %s | param1: %s | param2: %s\n",
                                            comando.c_str(),
                                            param1.c_str(),
                                            param2.c_str()
                                            );Serial.print(debug_buff);};
        param1.toCharArray(buff, NBUF);
        pinVal=atoi(buff);
        
        delayMicro=(param2=="u");
        
        if (delayMicro)  Tmicro=pinVal; else  Tmilli=pinVal ;
                                            
        //Serial.print("ok");
        msg("0");
        msgEnd();

      } // END DELAY
      
      
      
      
      
      
      
      // Si el comando es BINARY **************************************************************************
      else if(comando.substring(0,6)=="binary" || comando.substring(0,6)=="Binary" || comando.substring(0,6)=="BINARY"){
        param1=buffer.substring(esp1+1);
        pinStr=param1.substring(0);
        pinStr.toCharArray(buff, NBUF);
        binary_mode=atoi(buff);
        if(binary_mode>0) {
          debug_mode=0;
          msg("9");
          msgEnd();
          msg("Bynary Mode ON");
        }else{
          msg("Bynary Mode OFF");
        }
        msgEnd();
      }
      
      
      
      
      
      
      // Si el comando es VREF **************************************************************************
      else if(comando.substring(0,4)=="vref" || comando.substring(0,4)=="Vref" || comando.substring(0,4)=="VREF"){
        param1=buffer.substring(esp1+1);
        pinStr=param1.substring(0);
        pinStr.toCharArray(buff, NBUF);
        Vref=atoi(buff);
        
        if(Vref==1) {
            analogReference(INTERNAL1V1);
        }else if(Vref==2){
            analogReference(INTERNAL2V56);
        }else if(Vref==9){
            analogReference(EXTERNAL);
        }else{
            Vref=5;
            analogReference(DEFAULT);
        }
        
        msg("9");
        msgEnd();
        msg("Vref seteado en: ");
        Serial.print(Vref);
        msgEnd();
      }
      
      
      
      
      
      
      
      // Si el comando es CURV2 ****************************************************************************
      else if(comando.substring(0,5)=="curv2" || comando.substring(0,5)=="Curv2" || comando.substring(0,5)=="CURV2"){
        
        esp2=buffer.indexOf(' ',esp1+1);
        
        // param1 es lo que esta entre los dos espacios
        param1=buffer.substring(esp1+1,esp2);
        // param2 es lo que esta despues del segundo espacio
        param2=buffer.substring(esp2+1);
        
        pinStr=param1.substring(1);
        pinStr.toCharArray(buff, NBUF);
        pinNum=atoi(buff);
        
        pinStr=param2.substring(1);
        pinStr.toCharArray(buff, NBUF);
        pinNum2=atoi(buff);
        

        if(param1.substring(0,1)=="A" && param2.substring(0,1)=="D"){
            if( pinNum>=0 && pinNum<4 ){
                pinNum+=54;
            }else pinNum=54;
            
            if( !(pinNum2>=24 && pinNum2<30 )){
                pinNum2=24;
            }
            
            if(trigger_mode){  // Si tengo el trigger activado, espero a que triggeree
                noInterrupts();
                trigFlag=false;
                interrupts();
                
                time0=millis();
                while(!trigFlag) if(millis()-time0>3000) trigFlag=true;
            }
            
            
            if(debug_mode!=0) time0=millis();
            
            utime0=micros();
            // noInterrupts();  // -- Habilitar solo si no hay instruccion "delay"
            for(int i=0;i<NN;i++){
              curv0[i]=(unsigned int)analogRead(pinNum); // Leemos el analogico
              curvT[i]=(unsigned int)(micros()-time0);                         // Registramos el tiempo
              curvD[i]=digitalRead(pinNum2);             // Leemos el digital
              delayMicro ? delayMicroseconds(Tmicro) : delay(Tmilli);
            }
            //interrupts();     // -- Habilitar solo si no hay instruccion "delay"
            
            if(debug_mode!=0) time1=millis();
            
            //for(int i=0;i<NN;i++) curvT[i]=curvT[i]-utime0;  // Restamos el tiempo inicial
            
            delay(50);
            
            
            
            if(binary_mode>0){
                msg("2");
                msgEnd();
                Serial.print(3*NN);
                msgEnd();
                
                for(int i=0;i<NN;i++){
                  Serial.write( lowByte(curv0[i]));
                  Serial.write(highByte(curv0[i]));
                }
                for(int i=0;i<NN;i++){
                  Serial.write( lowByte(curvD[i]));
                  Serial.write(highByte(curvD[i]));
                }
                for(int i=0;i<NN;i++){
                  Serial.write( lowByte((unsigned int)curvT[i]));
                  Serial.write(highByte((unsigned int)curvT[i]));
                  //Serial.write( lowByte(curvT[i])>>8);
                  //Serial.write(highByte(curvT[i])>>8);
                }
                
            }else{
                msg("[");
                for(int i=0;i<NN;i++){
                    if(i%100==0) msg(" ...\n");
                    Serial.print(curv0[i]);
                    if(i<NN-1) msg(", ");
                }
                msg(" ];\n");
                
                msgEnd();
                
                msg("[");
                for(int i=0;i<NN;i++){
                    if(i%100==0) msg(" ...\n");
                    Serial.print(curvD[i]);
                    if(i<NN-1) msg(", ");
                }
                msg(" ];\n");
                
                msgEnd();
                
                msg("[");
                for(int i=0;i<NN;i++){
                    if(i%100==0) msg(" ...\n");
                    Serial.print(curvT[i]);
                    if(i<NN-1) msg(", ");
                }
                msg(" ];\n");
            }
            
            msgEnd();
            
            if(debug_mode!=0){debug_len=sprintf(debug_buff,"\nTIME: %d\n",time1-time0);Serial.print(debug_buff);msgEnd();};
    
                
        }
        
        
       
      }// END CURV2 ------------------
      
      
      
      
      // Si el comando es CURV ****************************************************************************
      else if(comando.substring(0,4)=="curv" || comando.substring(0,4)=="Curv" || comando.substring(0,4)=="CURV"){
        
        param1=buffer.substring(esp1+1);
        pinStr=param1.substring(1);
        pinStr.toCharArray(buff, NBUF);
        pinNum=atoi(buff);

        if(param1.substring(0,1)=="A"){
            if( pinNum>=0 && pinNum<4 ){
                pinNum+=54;
            }else pinNum=54;
            
            
          
            if(trigger_mode){  // Si tengo el trigger activado, espero a que triggeree
                //msg("trig: "); Serial.println(trigFlag);
                
                noInterrupts();
                trigFlag=false;
                interrupts();
                //msg("trig: "); Serial.println(trigFlag);
                
                time0=millis();
                while(!trigFlag) if(millis()-time0>3000) trigFlag=true;
                
                //msg("trig: "); Serial.println(trigFlag);
            }
          
            if(debug_mode!=0) time0=millis();
            
            utime0=micros();
            // noInterrupts();  // -- Habilitar solo si no hay instruccion "delay"
            for(int i=0;i<NN;i++){
              curv0[i]=(unsigned int)analogRead(pinNum); // Leemos el analogico
              curvT[i]=(unsigned int)micros();                         // Registramos el tiempo
              delayMicro ? delayMicroseconds(Tmicro) : delay(Tmilli);
            }
            //interrupts();     // -- Habilitar solo si no hay instruccion "delay"
            
            
            if(debug_mode!=0) time1=millis();
            delay(50);
            
            if(binary_mode>0){
                msg("2");
                msgEnd();
                Serial.print(2*NN);
                msgEnd();
                
                for(int i=0;i<NN;i++){
                  Serial.write( lowByte(curv0[i]));
                  Serial.write(highByte(curv0[i]));
                }
                
                for(int i=0;i<NN;i++){
                  Serial.write( lowByte((unsigned int)curvT[i]));
                  Serial.write(highByte((unsigned int)curvT[i]));
                  //Serial.write( lowByte(curvT[i])>>8);
                  //Serial.write(highByte(curvT[i])>>8);
                }
                
            }else{
                msg("[");
                for(int i=0;i<NN;i++){
                    if(i%100==0) msg(" ...\n");
                    Serial.print(curv0[i]);
                    if(i<NN-1) msg(", ");
                }
                msg(" ];\n");
                
                msgEnd();
                
                msg("[");
                for(int i=0;i<NN;i++){
                    if(i%100==0) msg(" ...\n");
                    Serial.print(curvT[i]);
                    if(i<NN-1) msg(", ");
                }
                msg(" ];\n");
                
            }

            if(debug_mode!=0){debug_len=sprintf(debug_buff,"\nTIME: %d\n",time1-time0);Serial.print(debug_buff);};

            msgEnd();
            
        } // Fin A
      } // END CURV -----------------------
      

      
    } // Fin procesamiento de comandos
    
    
    // Limpio variables de control
    buffer="";
    charBuff=' ';

  }
}




void msgEnd(){
  if(binary_mode>0){ 
    Serial.write(EOT); 
  }else{ 
    Serial.write('\n'); 
  }
}

void msg(char* txt){
    Serial.write(txt); 
}



// http://provideyourown.com/2012/secret-arduino-voltmeter-measure-battery-voltage/
/*
long readVcc() {
  // Read 1.1V reference against AVcc
  // set the reference to Vcc and the measurement to the internal 1.1V reference
  #if defined(__AVR_ATmega32U4__) || defined(__AVR_ATmega1280__) || defined(__AVR_ATmega2560__)
    ADMUX = _BV(REFS0) | _BV(MUX4) | _BV(MUX3) | _BV(MUX2) | _BV(MUX1);
  #elif defined (__AVR_ATtiny24__) || defined(__AVR_ATtiny44__) || defined(__AVR_ATtiny84__)
    ADMUX = _BV(MUX5) | _BV(MUX0);
  #elif defined (__AVR_ATtiny25__) || defined(__AVR_ATtiny45__) || defined(__AVR_ATtiny85__)
    ADMUX = _BV(MUX3) | _BV(MUX2);
  #else
    ADMUX = _BV(REFS0) | _BV(MUX3) | _BV(MUX2) | _BV(MUX1);
  #endif  
 
  delay(2); // Wait for Vref to settle
  ADCSRA |= _BV(ADSC); // Start conversion
  while (bit_is_set(ADCSRA,ADSC)); // measuring
 
  uint8_t low  = ADCL; // must read ADCL first - it then locks ADCH  
  uint8_t high = ADCH; // unlocks both
 
  long result = (high<<8) | low;
 
  result = 1125300L / result; // Calculate Vcc (in mV); 1125300 = 1.1*1023*1000
  return result; // Vcc in millivolts
}*/
