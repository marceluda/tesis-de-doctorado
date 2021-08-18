% Leer hasta el caracter EOT 0x4

function [yy,tt]=curvA0(s)
    datos=cmd(s,'curv A0');
    datos=reshape(datos,numel(datos)/2,2);
    vref=cmd(s,'get vref');
    
    if vref==1
        vref=1.1;
    end
    if vref==2
        vref=2.56;
    end
    if vref==9
        %vref=0.394;
        vref=1023;
    end
    
    y=datos(:,1);
    t=datos(:,2);
    
    yy=y*vref/1023;
    
    tt(1)=t(1);
    nmul=0;
    for i=2:numel(t)
        if t(i)<t(i-1)
            nmul=nmul+1;
        end
        tt(i)=t(i)+ (2^16-1)*nmul;
    end
    
    tt=tt-tt(1);
    tt=tt/1e6;
end

