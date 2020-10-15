%Tha ypologisw to spatial histogram gia olh thn BUwithPCA
%Tha bgazw mia domh M me fields->Matrix,vector,label
%
%Tha metatopizw ta shmeia ws pros to barikentro
%Kai tha vriskw thn aktina ths sfairas(Tha einai h megisth apostash
%ths vasis
%H AN2 periexei tis metatopismenes sintetagmenes
AN2=struct();
AN2=AN;
for i=1:2500
    px=mean(AN{1,i}.X(:));
    py=mean(AN{1,i}.Y(:));
    pz=mean(AN{1,i}.Z(:));
    AN2{i}.X(:)=AN2{i}.X(:)-px;
    AN2{i}.Y(:)=AN2{i}.Y(:)-py;
    AN2{i}.Z(:)=AN2{i}.Z(:)-pz;
end
r1=[];
for i=1:2500
    [theta,phi,r]=cart2sph(AN2{i}.X(:),AN2{i}.Y(:),AN2{i}.Z(:));    
    r1(end+1)=max(r);
end
MAX_r=max(r1);
M=struct();
M.Matrix=[];
for i=1:2500
    [Matrix]=cartesian_to_spherical_rep(AN2{1,i},MAX_r,10,10,10);
    M.Matrix{i}=Matrix;
end
% Tha metatrepw se dianisma to kathe spatial histogram
% gia na to perasw apo ta SVM. Kathe cell ths V exei to histograma se 
%dianisma
V=struct();
V.Vector=[];
m=size(M.Matrix{1});
n=prod(m);
for i=1:2500
    V.Vector{i}=reshape(M.Matrix{i},n,1);
end
M.Vector=[];
M.Vector=V.Vector;
%
%Labeling
%Gia na dhmiourgisw ta stings twn label
%Tha exw p.x. A|1|1|1-->1os actor,1o sinaisthma->angry,entash:1
%1-100 actors
%1-7 sinaisthmata->1:angry,2:disg.,3:fear,4:happy,5:neutr.,6:sad,7:surpr.
%1-4 entash gia to sunaisthma (1,2,3,4,6,7) gia to sunaisthma (5) mono 0
s1='A';
s2='|';
s3='0';
%
s=struct();
s.s1=[];
for i=1:100   %actor
    for j=1:7 %sinaisthma
        if j==5 %gia to neutral
            name=[s1,s2,num2str(i),s2,num2str(j),s2,num2str(s3)]; 
            s.s1{end+1}=name;
        else 
            for k=1:4  %gia tin entash twn sinaistimatwn ektos tou neutral
                name=[s1,s2,num2str(i),s2,num2str(j),s2,num2str(k)]; 
                s.s1{end+1}=name;
            end
        end
    end
end
M.label=[];
M.label=s.s1;

