%Cross_validation_leave_one_out
%Tha ylopoihsw to cross_validation_leave_one_out gia na to efarmosw sta svm
%gia to spatial histogram
%Tha afinw mia eikona eksw kathe fora kai sto telos tha kanw test me thn
%eikona auth kai training me tis ypoloipes
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
k=1:size(Matrix,2);
label_test=zeros(k(end),1);
CV=struct();
CV.LOO_train=[];
CV.LOO_test=[];
CV.LOO_label_train=[];
CV.LOO_label_test=[];
for i=1:size(Matrix,2)
    test=Matrix(:,i);
    CV.LOO_test{i}=test;
    
    Matrix2=Matrix;
    Matrix2(:,i)=[];
    training=Matrix2;
    CV.LOO_train{i}=training;
      
    CV.LOO_label_test{i}=i;
    
    m=k;
    m(i)=[];
    label_training=m';
    CV.LOO_label_train{i}=label_training;    
end