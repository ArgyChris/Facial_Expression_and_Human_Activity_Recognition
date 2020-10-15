function [Matrix,phi,theta,r]=cartesian_to_spherical_rep(AN,max_r,user_entry_gia_r,user_entry_gia_elevation,user_entry_gia_azimuth)

%Tha metatrepw tis 3Dcartesianes sintetagmenes se 3Dshperikes
%sintentagmenes
%Estw: p=(x,y,z)
%sphera=(r,theta,phi)
%r=sqrt(x^2+y^2+z^2)
%theta=arccos(z/r)
%To phi thelei ligo psaksimo phi=atan2(y,x)
%To phi(eleveation) bgainei -pi/2:pi/2
%Mporoume na xrisimopoihsoume cart2sph (ta apotelesmata einai se rad)
%tha prepei na metatopisoume tis sintetagmenes me vash to Barikentro(einai
%o mesos oros twn sintetagmenwn)
%teliko=X-barikentro
%To max_r einai h megisth apostash
[theta,phi,r]=metatroph(AN);
p=length(theta);
%Gia to range
% user_entry_gia_r=input('Eisagetai se poses aktines thelete na to xwrisetai:');
% user_entry_gia_r=fragments;
s=max_r/user_entry_gia_r;
parametros_r=fix(r/s)+1;
%Gia to elevation
% user_entry_gia_elevation=input('Eisagetai se posa height thelete na to xwrisetai:');
% user_entry_gia_elevation=fragments;
l=180/user_entry_gia_elevation;
new_phi=phi+90;
parametros_phi=fix(new_phi/l)+1;
%Gia to azimuthio
% user_entry_gia_azimuth=input('Eisagetai se posa sector thelete na to xwrisetai:');
% user_entry_gia_azimuth=fragments;
m=360/user_entry_gia_azimuth;
parametros_theta=zeros(length(theta),1);
for k=1:length(theta)
    if theta(k)>0
        parametros_theta(k)=fix(theta(k)/m)+1;
    else
        n=360-abs(theta(k));
        parametros_theta(k)=fix(n/m)+1;
    end
end

%Gia to teliko spatial histogram
Matrix=zeros(user_entry_gia_azimuth,user_entry_gia_elevation,user_entry_gia_r+1);
for i=1:p
    var1=parametros_theta(i);
    var2=parametros_phi(i);
    var3=parametros_r(i);
    Matrix(var1,var2,var3)=Matrix(var1,var2,var3)+1;
end

end
function [theta,phi,r]=metatroph(AN)
    p=length(AN.X);
    theta=zeros(p,1);
    phi=zeros(p,1);
    r=zeros(p,1);
    for i=1:p
        [theta(i),phi(i),r(i)]=cart2sph(AN.X(i),AN.Y(i),AN.Z(i));
    end
    theta=theta*(180/pi);
    phi=phi*(180/pi);
end

