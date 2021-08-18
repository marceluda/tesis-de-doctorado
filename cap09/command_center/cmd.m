% escribir un comando y leer el resultado

function [salida, control, NN]=cmd(s,txt)
    salida='error';
    if double(txt(end))~=10
        txt=[txt, 10];
    end
    
    escribir(s,txt);
    control=str2num(leer(s));
    
    if control==0
        salida='ok'
    elseif control==1
        salida=fread(s,1,'uint16');
    elseif control==2
        NN=str2num(leer(s));
        salida=fread(s,NN,'uint16');
    elseif control==9
        salida=leer(s);
    end

    pause(0.05)
    
    while s.BytesAvailable >0
        leer(s);
    end
end