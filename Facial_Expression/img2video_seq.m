%Creates a video a sequence of images
%The script resizes the images

#cd C:\users\achristod\sequence\000
mov1=avifile('motion_ver4.avi');
mov1.quality=100
d=dir;
m=length(d)-2;
for(i=3:m)
    I=imread(d(i).name);
    A=imresize(I,0.5);
    mov1=addframe(mov1,A);
end
mov1 = close(mov1);

