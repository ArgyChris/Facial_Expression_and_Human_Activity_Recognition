
%Human segmentation using the following paper:

% A. Christodoulidis, K.K. Delibasis and I. Maglogiannis, "Near real-time human silhouette
% and movement detection in indoor enviroments using fixed cameras" PETRA
% 2012.

%1) Background modelling by approximated median filtering
%2) Shadow detection by color information
%3) Human motion estimation by computing the velocity of the human
%--------------------------------------------------------------------------

% load movie
load movie2.mat

%Create the background model by assigning the first frame of the video.
%We work in the HSI chromatic domain because shadow suppression algorithm works in
%this domain 
ref_bg=rgb2hsi(movie2(1).cdata);  %HSI conversion using (equation 4)

%We assing the final segmentation
final_segmentation=zeros(size(movie2(1).cdata,1),size(movie2(1).cdata,2),size(movie2,2));

%Parameters for the shadow suppression (equation 5)
figure;
alpha=0.7;
beta=0.85;
Ts=0.5;
Th=0.8;

%Setting the parameters for the human motion detection, we set a bounding
%box where we search for the human
num_lin=size(movie2(1).cdata,1);
num_col=size(movie2(1).cdata,2);
minI=1;
maxI=num_lin;
minJ=1;
maxJ=num_col;
pad=30;
threshold=0.15;
SE=strel('square', 3);
index_center_of_massX=[];
index_center_of_massY=[];
 
%Time metrics
total_time=[];
bg_time=[];
sd_time=[];
CM_time=[];

%Read every frame
for i=2:200
    tic
    I2=rgb2hsi(movie2(i).cdata); %change the chromatic space
    
    %For each subsequent frame, if a pixel has a value greater (less) than the
    %corresponding pixel in the background model, then the value of this pixel
    %in the background model is increased (decreased) by one:
    %Algorithm: Background modelling in the paper
    %to speed up we update in an area (+30 pixels) around the human, at
    %every iteration we recompute the minI, maxI (see lines: 98-101 and 125-128)
     for k=max(1,minI-pad):min(num_lin,maxI+pad) 
        for l=max(1,minJ-pad):min(num_col,maxJ+pad)
            
            if I2(k,l,3)>ref_bg(k,l,3)
                ref_bg(k,l,3)=ref_bg(k,l,3)+1/255;
            elseif I2(k,l,3)<ref_bg(k,l,3) && ref_bg(k,l,3)>0
                ref_bg(k,l,3)=ref_bg(k,l,3)-1/255;
            else
                %We do nothing here, because we have 2 options
            end
            
        end
     end

%     ch_I2=I2(:,:,3);
%     ch_ref_bg=ref_bg(:,:,3);
%     index_high=find(ch_I2>ch_ref_bg);
%     ch_ref_bg(index_high)=ch_ref_bg(index_high)+1/255;
%     index_low=find(ch_I2>ch_ref_bg & ch_ref_bg>0);
%     ch_ref_bg(index_low)=ch_ref_bg(index_low)-1/255;
%     ref_bg(:,:,3)=ch_ref_bg;

    bg_time(end+1)=toc;
    
    Im=double(I2);
    Bg=double(ref_bg);
    
    %If we have enough movement, change in more than 50 pixels, apply the
    %shadow/human detection
    I=abs(Im(:,:,3)-Bg(:,:,3)); %Background subtraction here
    I3=(I>threshold);
    
    ind_pixels=find(I3>0);
    if size(ind_pixels,1)>50
        
        %Find the bounding box of the movment area
        [Iind,Jind]=ind2sub(size(I3),ind_pixels);
        minI=min(Iind);
        maxI=max(Iind);
        minJ=min(Jind);
        maxJ=max(Jind);
        rect=[minJ minI maxJ-minJ maxI-minI];
        Im_cropped=imcrop(Im,rect);
        Bg_cropped=imcrop(Bg,rect);
        
        %Apply the shadow suppression (equation 5) and
        %Algorithm:Segmentation of Human Silhouette
        ratio1=Im_cropped(:,:,3)./Bg_cropped(:,:,3);
        diff1=abs(Im_cropped(:,:,2)-Bg_cropped(:,:,2));
        diff2=min(abs(Im_cropped(:,:,1)-Bg_cropped(:,:,1)),360-abs(Im_cropped(:,:,1)-Bg_cropped(:,:,1)));
        Shadow=(ratio1>=alpha).*(ratio1<=beta).*(diff1<=Ts).*(diff2<=Th);
        
        sd_time(end+1)=toc;
        Shadow_recombined=zeros(size(I3));
        Shadow_recombined(minI:maxI,minJ:maxJ)=Shadow;
        
        %Remove the shadow
        I4=I3.*(Shadow_recombined==0);
        
        %Human motion estimation part
        I5=imerode(I4,SE);
        I6=zeros(size(I5));
        ind=find(I5>0);
        [Iind,Jind]=ind2sub(size(I5),ind);
        minI=min(Iind);
        maxI=max(Iind);
        minJ=min(Jind);
        maxJ=max(Jind);
        meanIind=mean(Iind);
        meanJind=mean(Jind);
        v_nonzeroX=[];
        v_nonzeroY=[];
        %Find the center of mass of the object  
        for v=1:length(Iind);
            %retain the on pixels if there are maximum 150 pixels away from the
            %center of mass
            if(abs(Iind(v)-meanIind)+abs(Jind(v)-meanJind)<150)
                I6(Iind(v),Jind(v))=1;
                v_nonzeroX(end+1)=Iind(v);
                v_nonzeroY(end+1)=Jind(v);
            else
                I6(Iind(v),Jind(v))=0;
            end
        end
        newmeanIind=mean(v_nonzeroX); 
        newmeanJind=mean(v_nonzeroY);
        newminI=min(v_nonzeroX);
        newmaxI=max(v_nonzeroX);
        newminJ=min(v_nonzeroY);
        newmaxJ=max(v_nonzeroY);
        CM_time(end+1)=toc;
        imshow(I6);title(['No. of frame:',num2str(i)]);
        
        final_segmentation(:,:,i)=I6;
        
        drawnow
        hold on;
        %Plot the Center of mass in the figure;
        if ~isnan(newmeanJind)
            plot(newmeanJind,newmeanIind,'*r');
            drawnow        
            vx=[newminJ,newmaxJ,newmaxJ,newminJ,newminJ];
            vy=[newminI,newminI,newmaxI,newmaxI,newminI];
            plot(vx,vy);
            drawnow        
        end
%         hold off;
        index_center_of_massX(end+1)=newmeanIind;
        index_center_of_massY(end+1)=newmeanJind;
        
        minI=newminI;
        minJ=newminJ;
        maxI=newmaxI;
        maxJ=newmaxJ;
    end
   
        total_time(end+1)=toc;
end
figure; plot(1:length(total_time),total_time); title(['total time (sec)'])
figure; plot(1:length(bg_time),bg_time); title(['background subtraction time (sec)'])
figure; plot(1:length(sd_time),sd_time); title(['shadow suppression time (sec)'])
figure; plot(1:length(CM_time),CM_time); title(['center of mass time (sec)'])