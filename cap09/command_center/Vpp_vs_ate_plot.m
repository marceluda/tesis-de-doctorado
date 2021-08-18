
load vpp_vs_ate.mat n_ate Vpp VppErr Vmin VminErr

plt=errorbar(n_ate,Vpp,VppErr)
gg=gca() 
set(gg,'YScale', 'log')
xlabel 'n ATE'
ylabel 'mVpp'
xlim([-0.5,15.5])
set(gg,'LineWidth',0.3)
set(plt,'LineWidth',1.5)
set(gg,'XTick',0:15)
grid()




for i=0:8:255
    cmd(s,['set P10 ', num2str(i)])
    disp(i)
    pause(2)
end





% Rutina para leer la frecuencia

vtek=vtek_preparar();
[tt ch1]=vtek_qch([1],vtek);

TT=(max(tt-min(tt)))/(numel(find(diff(ch1-mean(ch1)>0)))/2)

param=[max(ch1)-min(ch1), TT, 0, min(ch1)]

[f1,p1,cvg1,iter1,corp1,covp1]=leasqr(tt,ch1,param,'fit_func');


ig=[]
Tg=[]
Tvar=[]

for i=15:8:255
    cmd(s,['set P10 ',num2str(i)]);
    pause(10)
    [tt ch1]=vtek_qch([1],vtek);
    disp(i)
    pause(0.2)
    TT=(max(tt-min(tt)))/(numel(find(diff(smooth(ch1)-mean(ch1)>0)))/2);
    param=[max(ch1)-min(ch1), TT, 0, min(ch1)];
    [f1,p1,cvg1,iter1,corp1,covp1]=leasqr(tt,ch1,param,'fit_func');
    plot(tt,ch1,tt,f1)
    Tg=[Tg, p1(2)];
    Tvar=[Tvar, covp1(2,2)];
    ig=[ig, i];
    pause(0.55)
end
    


% plot(ig,Tg,'.-')


Vcc=4.87
% 
% plot(ig(3:end),Tg(3:end),'.-')
% grid()
% xlabel('n_{P10}')
% ylabel('periodo [s]')
% 




[aa,f1,f2]=plotyy(ig(3:end),Tg(3:end),ig(3:end),1./Tg(3:end))
set(f1,'Marker','.')
set(f2,'Marker','.')
set(aa(2),'XAxisLocation', 'top')

yL=get(aa(1),'YLim');
set(aa(1),'YTick',linspace(yL(1),yL(2),11))

yL=get(aa(2),'YLim');
set(aa(2),'YTick',linspace(yL(1),yL(2),11))


xt=(round(100*get(aa(1),'XTick')/255*Vcc)/100);
xt(find(xt>Vcc))=Vcc;
set(aa(2),'XTickLabel',xt)

ylabel(aa(1),'Periodo [s]')
xlabel(aa(1),'n_{P10}')

ylabel(aa(2),'Frecuencia [Hz]')
xlabel(aa(2),'V_{P10}')

grid()

% 
% a1=gca();
% a2=axes('position',get(a1,'position'));
% set(a2,'Color','none')
% set(a2,'XAxisLocation','top')
% set(a2,'YAxisLocation','right')
% 
% set(a2,'XLim',get(a1,'XLim'))
% set(a2,'XTick',get(a1,'XTick'))
% 
% set(a2,'YLim',sort(1./get(a1,'YLim')))
% set(a2,'YTick',sort(1./get(a1,'YTick')))


