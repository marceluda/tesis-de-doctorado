% Leer hasta el caracter EOT 0x4

function [txt]=leer(s)
    tic();
    sBuffer='';
    charBuff=' ';

    while charBuff~=4
        if s.BytesAvailable >0 
            charBuff=fread(s,1);
            if charBuff~=4
                sBuffer=[sBuffer,charBuff];
            end
        end
        if toc()>15
            charBuff=4;
            disp('timeout')
        end
    end
    txt=sBuffer;
end

