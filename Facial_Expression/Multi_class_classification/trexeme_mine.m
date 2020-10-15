%Tha kanw multi-class classification. %Tha prepei na to treksw me to Matlab
%7.1.0
%Tha bazw san training set 99 eikones(actors) apo kathe class/
%1 eikona(actor) apo kathe class gia test
%Training:training samples, NxM, (a column of row vectors) M:size_of_vector
%N:number_of_vector
%Labels:labels of training samples, Mx1, (a column vector) 
%Tha prepei na ta exw mazi Training+Labels (sindemena) ta labels sto telos
%mia stili
%To idio kai gia to Test-> alla tha einai mia eikona
%

% clear 
N=7; %arithmos klasewn
mine=[];
mine_actual=[];
confusion_matrix = zeros(N);
tic                  
%     load imagesTrain.mat;
% 	data=train_data_apo_dnmf;
% 	train_data=double(data(:,1:end-1));
%     labels_train=double(data(:,end));
train_data=Treksimo1.LOO_train{1};
labels_train=Treksimo1.LOO_label_train{1};
%  clear data;
%   load imagesTest.mat;
%  data=test_data_apo_dnmf;
%  test_data=double(data(:,1:end-1));
%  labels_test=double(data(:,end));
test_data=Treksimo1.LOO_test{1};
labels_test=Treksimo1.LOO_label_test{1};
%Nc=[164 170 195 355 245 397 1560];
%Nc=[35 37 54 89 66 71 352];
%Nc=[2486 4376]; %Tsapanos Frontal/non Frontal
% Nc=[57 55 56 85 53 57 239];
Nc=[99 99 99 99 99 99 99]; 
%Nc=[2810 2810 2810 2810 2810 2810 2810 2810];%edw grafeis apo posa deigmata apoteleitai h kathe klash sto training 

%%%%San sxolio tin kanonikopoihsh%%Giati den mou leitourgei
%nc=Nc;
% Training=train_data';%(1:end-1,:);
% sampleMean                              = mean(Training,2);
% [m n]                                   = size(Training);
% Sw                                      = zeros(m,m);
% Sb                                      = zeros(m,m);
% for i = 1:N
%     index_s                             = sum(nc(1:(i-1)));  
%     index                               = index_s+1:index_s+nc(i);   
%     classMean                           = mean(Training(:,index),2);
%     Sb                                  = Sb + nc(i)*(classMean-sampleMean)*(classMean-sampleMean)';
%     Sw                                  = Sw + (Training(:,index)-repmat(classMean,1,size(Training(:,index),2)))*(Training(:, index)-repmat(classMean,1,size(Training(:,index),2)))';
% end
 
% train_data=[train_data*real(Sw^(-1/2))     labels_train]; 
% test_data=[test_data*real(Sw^(-1/2))       labels_test ]; 

%Mi kanonikopoihmena giati einai sparse o pinakas
train_data=[train_data     labels_train]; 
test_data=[test_data       labels_test ]; 
% save final_train train_data  %auta bazeis sta svms ths c
% save final_test test_data   %auta bazeis sta svms ths c
%      %   
  train_labels=labels_train;
  test_labels=labels_test;
    %dhmiourgia pinakwn eksodwn tou  svm
    y_train = [ 2*(train_data(:,end) == 1)-1,   2*(train_data(:,end) == 2)-1, 2*(train_data(:,end) == 3)-1 ,   2*(train_data(:,end) == 4)-1 ,2*(train_data(:,end) == 5)-1 ,   2*(train_data(:,end) == 6)-1 ,   2*(train_data(:,end) == 7)-1 ];
    y_test = [ 2*(test_data(:,end) == 1)-1,   2*(test_data(:,end) ==2 )-1, 2*(test_data(:,end) == 3)-1 ,   2*(test_data(:,end) == 4)-1 , 2*(test_data(:,end) == 5)-1 ,   2*(test_data(:,end) == 6)-1 ,   2*(test_data(:,end) == 7)-1 ];
    train_data(:,end)=[];
    test_data(:,end)=[]; 
    k=N; 
    %kernel type-->typos tou kernel->polyonimal(degree)
    %to C einai enas sintelesths paei apo 1-10000 (den kserw ti akrivws
    %paizei me auth thn metavlith
    %to rbf den douleue sto tropo mou
    kernel =polynomial(1); 
  	%kernel = rbf(3);
	%kernel=linear;  
	C      = 15.0;
    tutor  = smosvctutor;   % this means we use the SMO training algorithm
    net    = dagsvm;        % this means we use the DAG-SVM algortihm to combine
	% the outputs of a number of 2-class networks
      net = train(net, tutor, train_data, y_train, C, kernel);%dhmiourgia svm 6 klasewn
    % generate confusion matrix   
    o = fwd(net, test_data);    %eisagoume sto svm ta test data
    [tmp1,Y] = max(y_test');
    [tmp2,O] = max(o');
    for i=1:k
        for j=1:k
            confusion_matrix(i,j) = length(find(O == i & Y == j))+confusion_matrix(i,j);      %dhmiourgia confusion_matrix
        end
    end
    mine=[mine;O'];
    mine_actual=[mine_actual;test_labels];
    confusion_matrix
    agreement = find(O == Y);
    accuracy = length(agreement)*100/length(y_test);%upologismos akribeias gia to dothen sunolo training kai testing data
      fprintf('H akribeia ths anagnwrishs einai ish me %f %',accuracy);
    fprintf('\nMean classification acc=%f',mean(diag(confusion_matrix)'./sum(confusion_matrix)));
train_data=[  train_data(:,1:end)      labels_train];%*real(Sw^(-1/2)))
test_data=[  test_data(:,1:end)        labels_test ]; %*real(Sw^(-1/2))*real(Sw^(-1/2)) 
% load Distances;