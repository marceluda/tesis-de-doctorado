% oscilloscopio arduino

%s=serial('/dev/ttyACM1');
s=serial('COM6');
set(s,'BaudRate',57600);
set(s,'InputBufferSize',4096);


% Nota: para que el MATLAB lea correctamente el puerto es necesario 
%       crear un archivo en el home folder donde abre el matlab
%       que en mi caso es /home/lolo
%
%       Archivo: java.opts
%       -Dgnu.io.rxtx.SerialPorts=/dev/ttyS0:/dev/ttyS1:/dev/USB0:/dev/ttyACM0:/dev/ttyACM1
%
disp('lolo no')

break;
disp('lolo')




fopen(s)
pause(2)
leer(s)
cmd(s,'binary 1')

cmd(s,'trig up')

cmd(s,'trig down')

cmd(s,'trig off')





leer(s)

escribir(s,'\n')


for i=1:20
    plot(cmd(s,'curv A0')*5/1024,'.-')
    pause(0.1)
end

escala=1.1
escala=5
escala=1024

cmd(s,'ate 15')

cmd(s,'set P9 115')

[yy,tt]=curvA0(s);
plot(tt,yy,'.-')
plot(tt,smooth(yy),'.-')

cmd(s,'vref 9')
cmd(s,'delay 250 u')


% t_start=cputime();
% yy=cmd(s,'curv A0');
% t_tot=cputime()-t_start;
% plot(t_tot*[1:length(yy)]/length(yy),yy*escala/1024,'.-')
% disp(['Vpp=',num2str(...
%     max(yy*escala/1024)-min(yy*escala/1024)...
%     ), '   mean=', num2str(...
% mean(yy*escala/1024)), '   std=',num2str(...
% std(yy*escala/1024))]);

%aa=[aa, mean(yy*escala/1024)];

cmd(s,'trig up')
cmd(s,'trig down')


cmd(s,'set P10 150')
cmd(s,'set P10 0')

cmd(s,'set P9 100')

cmd(s,'trig off')


cmd(s,'set D30 0')
cmd(s,'set D32 0')
cmd(s,'set D34 0')
cmd(s,'set D36 1')

cmd(s,'vref 5')
cmd(s,'vref 1')

plot(cmd(s,'curv A0')*5/1024,'.-')

cmd(s,'curv A0')

tic()
cmd(s,'curv2 A0 D26')
toc()

fclose(s)

