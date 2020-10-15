accuracy=[];
accuracy_teliko=[];
for p_poly=1:10
    for i_j=1:99
        N=7;
        mine=[];
        mine_actual=[];
        confusion_matrix = zeros(N);
        train_data=CV.LOO_train{i_j};
        labels_train=CV.LOO_label_train{i_j};
        test_data=CV.LOO_test{i_j};
        labels_test=CV.LOO_label_test{i_j};
        Nc=[99 99 99 99 99 99 99]; 
        train_data=[train_data  labels_train]; 
        test_data=[test_data    labels_test ]; 
        train_labels=labels_train;
        test_labels=labels_test;
        y_train = [ 2*(train_data(:,end) == 1)-1,   2*(train_data(:,end) == 2)-1, 2*(train_data(:,end) == 3)-1 ,   2*(train_data(:,end) == 4)-1 ,2*(train_data(:,end) == 5)-1 ,   2*(train_data(:,end) == 6)-1 ,   2*(train_data(:,end) == 7)-1 ];
        y_test = [ 2*(test_data(:,end) == 1)-1,   2*(test_data(:,end) ==2 )-1, 2*(test_data(:,end) == 3)-1 ,   2*(test_data(:,end) == 4)-1 , 2*(test_data(:,end) == 5)-1 ,   2*(test_data(:,end) == 6)-1 ,   2*(test_data(:,end) == 7)-1 ];
        train_data(:,end)=[];
        test_data(:,end)=[]; 
        k=N; 
%         kernel =polynomial(3);
        kernel =polynomial(p_poly); %Bgazei girw sto 45(ws 4 degree apo 5-10 14%)
%         kernel = rbf(p_poly); %Bgazei girw sto 14%
    %     kernel=linear;   %Bgazei girw sto 45%
        C      = 15.0;
        tutor  = smosvctutor;   % this means we use the SMO training algorithm
        net    = dagsvm;        % this means we use the DAG-SVM algortihm to combine
        net = train(net, tutor, train_data, y_train, C, kernel);   
        o = fwd(net, test_data);   
        [tmp1,Y] = max(y_test');
        [tmp2,O] = max(o');
        for i=1:k
            for j=1:k
                confusion_matrix(i,j) = length(find(O == i & Y == j))+confusion_matrix(i,j);      %dhmiourgia confusion_matrix
            end
        end
        mine=[mine;O'];
        mine_actual=[mine_actual;test_labels];
        agreement = find(O == Y);
        accuracy(i_j,p_poly) = length(agreement)*100/length(y_test);
%     accuracy(i_j) = length(agreement)*100/length(y_test);
    end
accuracy_teliko(p_poly)=mean(accuracy(:,p_poly));
end