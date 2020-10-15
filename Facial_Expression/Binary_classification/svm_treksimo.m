%ypoloipa apo to SVMTrain, SVMTest sta opoia mporw na xeiristw poio
%dieksodika tis parametrous
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%Kwdikas gia training-testing
function [expr1_true,expr2_true]=svm_treksimo(M)
%epilogh classewn gia svm kathws kai entashs tou sinaisthmatos
user_def_expression1=input('1o(angry-1,disgust-2,fear-3,happy-4,neutral-5,sad-6,suprise-7):');
user_def_expression2=input('2o(angry-1,disgust-2,fear-3,happy-4,neutral-5,sad-6,suprise-7):');
user_int_express=input('entash_expression(1-4),gia to neutral 0:');
%training ana 2 twn klassewn
[token,remain]=strtok(M.label,'|');
[token1,remain]=strtok(remain,'|');
[token2,remain]=strtok(remain,'|');
[token3,remain]=strtok(remain,'|');
t2=str2double(token2);
t3=str2double(token3);
ind1=find(t2==user_def_expression1 & t3==user_int_express);
ind2=find(t2==user_def_expression2 & t3==user_int_express);

% %sthn periptwsh pou sigrinw me to neutral (exei intensity_expression=0)
% ind2=find(t2==user_def_expression2 & t3==0);   

p=length(ind1);
Matrix1=[];Matrix2=[]; k1=[];k2=[];
for i=1:p
    k1=M.Vector{ind1(i)};k2=M.Vector{ind2(i)};
    Matrix1(:,end+1)=k1;Matrix2(:,end+1)=k2;
end
Matrix=[Matrix1,Matrix2];
l1=ones(1,p)*user_def_expression1; l2=ones(1,p)*user_def_expression2; 
label=[l1';l2'];
% label2=[l2';l1'];
%%%Dhmiourgia twn cross validate training and test set
% [CV]=cross_validation_leave_one_out(Matrix); %einai leave one mia eikona oxi actor
%Tha kane leave one actor out

[expr1_true,expr2_true]=Test_expressions(CV,user_def_expression1,user_def_expression2,label);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%Training
function [AlphaY, SVs, Bias, Parameters, nSV, nLabel] = SVMTrain(Samples, Labels, Parameters, Weight)
% Usages:
% [AlphaY, SVs, Bias, Parameters, nSV, nLabel] = SVMTrain(Samples, Labels)
% [AlphaY, SVs, Bias, Parameters, nSV, nLabel] = SVMTrain(Samples, Labels, Parameters)
% [AlphaY, SVs, Bias, Parameters, nSV, nLabel] = SVMTrain(Samples, Labels, Parameters, Weight)
%
% DESCRIPTION:
% Construct a SVM classifier. 
% In fact, This function is used to do the input parameter checking, and it calls a mex file, mexSVMTrain, to 
% implement the algorithm.
%                                   
% Inputs:
%    Samples    - training samples, MxN, (a row of column vectors);
%    Labels     - labels of training samples, 1xN, (a row vector);
%    Parameters - the paramters required by the training algorithm (a <=11-element row vector);
%     +------------------------------------------------------------------
%     |Kernel Type| Degree | Gamma | Coefficient | C |Cache size|epsilon| 
%     +------------------------------------------------------------------
%       ----------------------------------------------+
%       | SVM type | nu | loss toleration | shrinking |
%       ----------------------------------------------+
%            where Kernel Type: (default: 2) 
%                     0 --- Linear
%                     1 --- Polynomial: (Gamma*<X(:,i),X(:,j)>+Coefficient)^Degree
%                     2 --- RBF: (exp(-Gamma*|X(:,i)-X(:,j)|^2)) 
%                     3 --- Sigmoid: tanh(Gamma*<X(:,i),X(:,j)>+Coefficient)
%                  Degree: default 3
%                  Gamma: If the input value is zero, Gamma will be set defautly as
%                         1/(max_pattern_dimension) in the function. If the input
%                         value is non-zero, Gamma will remain unchanged in the 
%                         function. (default: 1)
%                  Coefficient: default 0
%                  C: Cost of constrain violation for C-SVC, epsilon-SVR, and nu-SVR (default 1)
%                  Cache Size: Space to hold the elements of K(<X(:,i),X(:,j)>) matrix (default 40MB)
%                  epsilon: tolerance of termination criterion (default: 0.001)
%                  SVM Type: (default: 0)
%                     0 --- c-SVC 
%                     1 --- nu-SVC
%                     2 --- one-class SVM
%                     3 --- epsilon-SVR 
%                     4 --- nu-SVR
%                  nu: nu of nu-SVC, one-class SVM, and nu-SVR (default: 0.5)
%                  loss tolerance: epsilon in loss function of epsilon-SVR (default: 0.1)
%                  shrinking: whether to use the shrinking heuristics, 0 or 1 (default: 1)
%    Weight     - a row vector or scalar, C of class i is weight(i)*C in C-SVC (default: all 1's);
%
% Outputs:  
%    AlphaY    - Alpha * Y, where Alpha is the non-zero Lagrange Coefficients, and
%                    Y is the corresponding Labels, (L-1) x sum(nSV);
%                All the AlphaYs are organized as follows: (pretty fuzzy !)
%      				classifier between class i and j: coefficients with
%			  	         i are in AlphaY(j-1, start_Pos_of_i:(start_Pos_of_i+1)-1),
%				         j are in AlphaY(i, start_Pos_of_j:(start_Pos_of_j+1)-1)
%    SVs       - Support Vectors. (Sample corresponding the non-zero Alpha), M x sum(nSV),
%                All the SVs are stored in the format as follows:
%                 [SVs from Class 1, SVs from Class 2, ... SVs from Class L];
%    Bias      - Bias of all the 2-class classifier(s), 1 x L*(L-1)/2;
%    Parameters -  Output parameters used in training;
%    nSV       -  numbers of SVs in each class, 1xL;
%    nLabel    -  Labels of each class, 1xL.
%
% By Junshui Ma, and Yi Zhao (02/15/2002)
%

if (nargin < 2 | nargin > 4)
   disp(' Error: Incorrect number of input variables.');
   help SVMTrain;
   return
end

if (nargin >= 3) 
    [prM prN]= size(Parameters);
    if (prM ~= 1 & prN~=1)
        disp(' Error: ''Parameters'' should be a row vector.');
        return
    elseif (prM~= 1)
        Parameters = Parameters';
        [prM prN]= size(Parameters);
    end
    if (Parameters(1)>3) & (Parameters(1) < 0)
        disp(' Error: this program only supports 4 types of kernel functions.');
        return
    end
    if (prN >=8)
        if (Parameters(8)>4) & (Parameters(8) <0)
           disp(' Error: this program only supports 5 types of SVMs.');
           return
        end
    end
    if (prN >=9)    
        if ((Parameters(8)==1) | (Parameters(8) == 2) | (Parameters(8) == 4)) & (Parameters(9) >= 1)
           disp(' Error: the nu for nu-SVC, one-class SVM, and nu-SVR should be less than 1 and bigger than 0');
           return
        end        
    end
end

[spM spN]=size(Samples);
[lbM lbN]=size(Labels);
if lbM ~= 1
   disp(' Error: ''Labels'' should be a row vector.');
   return
end
if spN ~= lbN
   disp(' Error: the number of training samples is different from that of their labels.');
   return
end

% call the mex file
if (nargin == 2)
    [AlphaY, SVs, Bias, Parameters, nSV, nLabel] = mexSVMTrain(Samples, Labels);
elseif (nargin == 3)
    [AlphaY, SVs, Bias, Parameters, nSV, nLabel] = mexSVMTrain(Samples, Labels, Parameters);
elseif (nargin == 4)
    [AlphaY, SVs, Bias, Parameters, nSV, nLabel] = mexSVMTrain(Samples, Labels, Parameters, Weight);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%Testing
function [ClassRate, DecisionValue, Ns, ConfMatrix, PreLabels]= SVMTest(Samples, Labels, AlphaY, SVs, Bias,Parameters, nSV, nLabel)
% Usages:
%  [ClassRate, DecisionValue, Ns, ConfMatrix, PreLabels]= SVMTest(Samples, Labels, AlphaY, SVs, Bias)
%  [ClassRate, DecisionValue, Ns, ConfMatrix, PreLabels]= SVMTest(Samples, Labels, AlphaY, SVs, Bias, Parameters)
%     Note that the above two formats are only valid for 2-class problem, it is implemented here to make this version 
%      to be compatabible with the previous version of OSU SVM ToolBox.
%  [ClassRate, DecisionValue, Ns, ConfMatrix, PreLabels]= SVMTest(Samples, Labels, AlphaY, SVs, Bias, Parameters, nSV, nLabel)
%
% DESCRIPTION:
%    Test the performance of a trained SVM classifier by a group of input patterns
%    with their true class labels given.
%    In fact, this function is used to do the input parameter checking, and it 
%    depends on a mex file, mexSVMClass, to implement the algorithm.
%
% Inputs:
%    Samples    - testing samples, MxN, (a row of column vectors);
%    Labels     - labels of testing samples, 1xN, (a row vector);
%    AlphaY    - Alpha * Y, where Alpha is the non-zero Lagrange Coefficients, and
%                    Y is the corresponding Labels, (L-1) x sum(nSV);
%                All the AlphaYs are organized as follows: (pretty fuzzy !)
%      				classifier between class i and j: coefficients with
%			  	         i are in AlphaY(j-1, start_Pos_of_i:(start_Pos_of_i+1)-1),
%				         j are in AlphaY(i, start_Pos_of_j:(start_Pos_of_j+1)-1)
%    SVs       - Support Vectors. (Sample corresponding the non-zero Alpha), M x sum(nSV),
%                All the SVs are stored in the format as follows:
%                 [SVs from Class 1, SVs from Class 2, ... SVs from Class L];
%    Bias      - Bias of all the 2-class classifier(s), 1 x L*(L-1)/2;
%    Parameters - the paramters required by the training algorithm (a <=11-element row vector);
%     +------------------------------------------------------------------
%     |Kernel Type| Degree | Gamma | Coefficient | C |Cache size|epsilon| 
%     +------------------------------------------------------------------
%       ----------------------------------------------+
%       | SVM type | nu | loss toleration | shrinking |
%       ----------------------------------------------+
%            where Kernel Type: (default: 2) 
%                     0 --- Linear
%                     1 --- Polynomial: (Gamma*<X(:,i),X(:,j)>+Coefficient)^Degree
%                     2 --- RBF: (exp(-Gamma*|X(:,i)-X(:,j)|^2)) 
%                     3 --- Sigmoid: tanh(Gamma*<X(:,i),X(:,j)>+Coefficient)
%                  Degree: default 3
%                  Gamma: If the input value is zero, Gamma will be set defautly as
%                         1/(max_pattern_dimension) in the function. If the input
%                         value is non-zero, Gamma will remain unchanged in the 
%                         function. (default: 0 or 1/M)
%                  Coefficient: default 0
%                  C: Cost of constrain violation for C-SVC, epsilon-SVR, and nu-SVR (default 1)
%                  Cache Size: Space to hold the elements of K(<X(:,i),X(:,j)>) matrix (default 40MB)
%                  epsilon: tolerance of termination criterion (default: 0.001)
%                  SVM Type: (default: 0)
%                     0 --- c-SVC 
%                     1 --- nu-SVC
%                     2 --- one-class SVM
%                     3 --- epsilon-SVR 
%                     4 --- nu-SVR
%                  nu: nu of nu-SVC, one-class SVM, and nu-SVR (default: 0.5)
%                  loss tolerance: epsilon in loss function of epsilon-SVR (default: 0.1)
%                  shrinking: whether to use the shrinking heuristics, 0 or 1 (default: 1)
%    nSV       -  numbers of SVs in each class, 1xL;
%    nLabel    -  Labels of each class, 1xL.
%
% Outputs:  
%    ClassRate      -  Classification rate, 1x1;
%    DecisionValue  -  the output of the decision function (only meaningful for 2-class problem), 1xN;
%    Ns             -  number of samples in each class, 1x(L+1), or 1xL;
%                       Note that the last element is for the Samples that are not in any
%                         classes in the training set.
%    ConfMatrix     -  Confusion Matrix, (L+1)x(L+1), or LxL, where ConfMatrix(i,j) = P(X in j| X in i);
%                       Note that when (L+1)x(L+1), the last row and the last column are for the Samples 
%                       that are not in any classes in the training set.
%    PreLabels      -  Predicated Labels, 1xN. 
%
% By Junshui Ma, and Yi Zhao (02/15/2002)
%

if (nargin < 5 | nargin > 8)
   disp(' Error: Incorrect number of input variables.');
   help SVMTest;
   return
end

[minLabel, I]=min(Labels);
[maxLabel, I]=max(Labels);
if ((minLabel ~= -1) | (maxLabel ~= 1))
    if (nargin < 8)
        disp(' Error: The sample labels are not in {-1,1}, However, you need to input ''nLabel'' to support speical labels.');
        return
    end
end
    

if (nargin >= 6) 
    [prM prN]= size(Parameters);
    if (prM ~= 1 & prN~=1)
        disp(' Error: ''Parameters'' should be a row vector.');
        return
    elseif (prM~= 1)
        Parameters = Parameters';
        [prM prN]= size(Parameters);
    end
    if (Parameters(1)>3) & (Parameters(1) < 0)
        disp(' Error: this program only supports 4 types of kernel functions.');
        return
    end
    if (prN >=8)
        if (Parameters(8)>4) & (Parameters(8) <0)
           disp(' Error: this program only supports 5 types of SVMs.');
           return
        end
    end
end



[alM alN] = size(AlphaY);
if (nargin <= 6)  
    [r c] = size(Bias);
    if (r~=1 | c~=1)
        disp(' Error: Your SVM classifier seems a multiclass classifier. However, you need to input ''nSV'' and ''nLabel'' to support multiclass problem.');
        return
    end    
    if (alM > 1)
        disp(' Error: Your SVM classifier seems a multiclass classifier. However, you need to input ''nSV'' and ''nLabel'' to support multiclass problem.');
        return
    end 
end

[spM spN]=size(Samples);
[svM svN]=size(SVs);

if svM ~= spM
   disp(' Error: ''SVs'' should have the same feature dimension as ''Samples''.');
   return;
end

if svN ~= alN
   disp(' Error: number of ''SVs'' should be the same as the colmun number of ''AlphaY''.');
   return;
end


% call the mex file
if (nargin == 5)
    [ClassRate, DecisionValue, Ns, ConfMatrix, PreLabels]= mexSVMClass(Samples, Labels, AlphaY, SVs, Bias);
elseif (nargin == 6)
    [ClassRate, DecisionValue, Ns, ConfMatrix, PreLabels]= mexSVMClass(Samples, Labels, AlphaY, SVs, Bias,Parameters);
elseif (nargin == 8)
    [ClassRate, DecisionValue, Ns, ConfMatrix, PreLabels]= mexSVMClass(Samples, Labels, AlphaY, SVs, Bias,Parameters, nSV, nLabel);
end

% if these is no extra class in the testing samples, 
% remove that last column and row in ConfMatrix
if (ConfMatrix(end, end) == 1) 
    ConfMatrix = ConfMatrix(1:end-1,1:end-1);
    Ns = Ns(1:end-1);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Leave-One-Out Cross-Validation
function [CV]=cross_validation_leave_one_out(Matrix)
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
    label_training=m;
    CV.LOO_label_train{i}=label_training;    
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [var1,var2]=Test_expressions(CV,expr1,expr2,label)
var1=0;var2=0;var3=0;
for i_r=1:200
%     Parameters=[1,3,1,0,1.999,40,0.001,0,0.9950,-0.0002,0];
    [AlphaY1, SVs1, Bias1, Parameters, nSV1, nLabel1]=SVMTrain(CV.LOO_train{i_r},label(CV.LOO_label_train{i_r})');
    Parameters(1)=1;Parameters(2)=5;
    [AlphaY, SVs, Bias, Parameters, nSV, nLabel]=SVMTrain(CV.LOO_train{i_r},label(CV.LOO_label_train{i_r})',Parameters);    
    %elegxos gia na dw an allazoun oi parametroi
%     [AlphaY1, SVs1, Bias1, Parameters_gia_test, nSV1, nLabel1]=SVMTrain(CV.LOO_train{i_r},label(CV.LOO_label_train{i_r})');
    [ClassRate, DecisionValue, Ns, ConfMatrix, PreLabels]=SVMTest(CV.LOO_test{i_r}, label(CV.LOO_label_test{i_r})', AlphaY, SVs, Bias, Parameters,nSV,nLabel );
    c=PreLabels;
    label(CV.LOO_label_test{i_r});
%     k=Parameters_gia_test(2:end);
    if PreLabels==expr1 && label(CV.LOO_label_test{i_r})==expr1
        var1=var1+1;
    end
    if PreLabels==expr2 && label(CV.LOO_label_test{i_r})==expr2
        var2=var2+1;
    end
%     %elegxos gia na dw an alazoun oi parametroi
%     if k~=Parameters(2:end)
%         var3=var3+1;
%     end
end
function [CV]=cross_validation_leave_one_person_out(M)
%Cross_validation_leave_one_actor_out
%Tha ylopoihsw to cross_validation_leave_one_actor_out gia na to efarmosw sta svm
%gia to spatial histogram
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



