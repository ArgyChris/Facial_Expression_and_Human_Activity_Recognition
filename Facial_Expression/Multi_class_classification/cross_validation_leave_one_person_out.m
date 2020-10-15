function [CV]=cross_validation_leave_one_person_out(M)
%Cross_validation_leave_one_actor_out
%Tha ylopoihsw to cross_validation_leave_one_actor_out gia na to efarmosw sta svm
%gia to spatial histogram O parakatw kwdikas trexei gia spatial histogram
%ws 10 tmhmatopoihseis
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CV=struct();
% CV.LOO_train=[];
% CV.LOO_test=[];
% CV.LOO_label_train=[];
% CV.LOO_label_test=[];
% [token,remain] =strtok(M.label,'|');
% [token1,remain]=strtok(remain,'|');
% [token2,remain]=strtok(remain,'|');
% [token3,remain]=strtok(remain,'|');
% t2=str2double(token2);
% t3=str2double(token3);
% ind1=find(t2==1 & t3==4);
% ind2=find(t2==2 & t3==4);
% ind3=find(t2==3 & t3==4);
% ind4=find(t2==4 & t3==4);
% ind5=find(t2==5 & t3==0);
% ind6=find(t2==6 & t3==4);
% ind7=find(t2==7 & t3==4);
% p=length(ind1);
% Matrix1=[];Matrix2=[];Matrix3=[];Matrix4=[];
% Matrix5=[];Matrix6=[];Matrix7=[];
% for i=1:p
%     Matrix1(i,:)=M.Vector{ind1(i)};
%     Matrix2(i,:)=M.Vector{ind2(i)};
%     Matrix3(i,:)=M.Vector{ind3(i)};
%     Matrix4(i,:)=M.Vector{ind4(i)};
%     Matrix5(i,:)=M.Vector{ind5(i)};
%     Matrix6(i,:)=M.Vector{ind6(i)};
%     Matrix7(i,:)=M.Vector{ind7(i)};
% end
% Matrix1_new=[];Matrix2_new=[];Matrix3_new=[];Matrix4_new=[];
% Matrix5_new=[];Matrix6_new=[];Matrix7_new=[];
% for i=1:99
%     test=[Matrix1(i,:);Matrix2(i,:);Matrix3(i,:);Matrix4(i,:);Matrix5(i,:);Matrix6(i,:);Matrix7(i,:)];
%     CV.LOO_test{i}=test;
%     Matrix1_new=Matrix1;
%     Matrix2_new=Matrix2;
%     Matrix3_new=Matrix3;
%     Matrix4_new=Matrix4;
%     Matrix5_new=Matrix5;
%     Matrix6_new=Matrix6;
%     Matrix7_new=Matrix7;
%     Matrix1_new(i,:)=[];
%     Matrix2_new(i,:)=[];
%     Matrix3_new(i,:)=[];
%     Matrix4_new(i,:)=[];
%     Matrix5_new(i,:)=[];
%     Matrix6_new(i,:)=[];
%     Matrix7_new(i,:)=[];
%     training=[Matrix1_new;Matrix2_new;Matrix3_new;Matrix4_new;Matrix5_new;Matrix6_new;Matrix7_new];
%     CV.LOO_train{i}=training;    
%     CV.LOO_label_train{i}=[ones(99,1);ones(99,1)*2;ones(99,1)*3;ones(99,1)*4;ones(99,1)*5;ones(99,1)*6;ones(99,1)*7];
%     CV.LOO_label_test{i}=[1;2;3;4;5;6;7];
% end

%Cross_validation_leave_one_actor_out
%Tha ylopoihsw to cross_validation_leave_one_actor_out gia na to efarmosw sta svm
%gia to spatial histogram O parakatw kwdikas trexei gia spatial histogram
%>10 tmhmatopoihseis giati exw provlima memory me to proigoumeno
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
CV=struct();
CV.LOO_train=[];
CV.LOO_test=[];
CV.LOO_label_train=[];
CV.LOO_label_test=[];
[token,remain] =strtok(M.label,'|');
[token1,remain]=strtok(remain,'|');
[token2,remain]=strtok(remain,'|');
[token3,remain]=strtok(remain,'|');
t2=str2double(token2);
t3=str2double(token3);
ind1=find(t2==1 & t3==4);
ind2=find(t2==2 & t3==4);
ind3=find(t2==3 & t3==4);
ind4=find(t2==4 & t3==4);
ind5=find(t2==5 & t3==0);
ind6=find(t2==6 & t3==4);
ind7=find(t2==7 & t3==4);
p=length(ind1);
Matrix1=[];Matrix2=[];Matrix3=[];Matrix4=[];
Matrix5=[];Matrix6=[];Matrix7=[];
for i=1:p
    Matrix1(i,:)=M.Vector{ind1(i)};
    Matrix2(i,:)=M.Vector{ind2(i)};
    Matrix3(i,:)=M.Vector{ind3(i)};
    Matrix4(i,:)=M.Vector{ind4(i)};
    Matrix5(i,:)=M.Vector{ind5(i)};
    Matrix6(i,:)=M.Vector{ind6(i)};
    Matrix7(i,:)=M.Vector{ind7(i)};
end
Matrix1_new=[];Matrix2_new=[];Matrix3_new=[];Matrix4_new=[];
Matrix5_new=[];Matrix6_new=[];Matrix7_new=[];
for i=1:99
    test=[Matrix1(i,:);Matrix2(i,:);Matrix3(i,:);Matrix4(i,:);Matrix5(i,:);Matrix6(i,:);Matrix7(i,:)];
    CV.LOO_test{i}=test;
    Matrix1_new=Matrix1;
    Matrix2_new=Matrix2;
    Matrix3_new=Matrix3;
    Matrix4_new=Matrix4;
    Matrix5_new=Matrix5;
    Matrix6_new=Matrix6;
    Matrix7_new=Matrix7;
    Matrix1_new(i,:)=[];
    Matrix2_new(i,:)=[];
    Matrix3_new(i,:)=[];
    Matrix4_new(i,:)=[];
    Matrix5_new(i,:)=[];
    Matrix6_new(i,:)=[];
    Matrix7_new(i,:)=[];
    training=[Matrix1_new;Matrix2_new;Matrix3_new;Matrix4_new;Matrix5_new;Matrix6_new;Matrix7_new];
    CV.LOO_train{i}=training;    
    CV.LOO_label_train{i}=[ones(99,1);ones(99,1)*2;ones(99,1)*3;ones(99,1)*4;ones(99,1)*5;ones(99,1)*6;ones(99,1)*7];
    CV.LOO_label_test{i}=[1;2;3;4;5;6;7];
end