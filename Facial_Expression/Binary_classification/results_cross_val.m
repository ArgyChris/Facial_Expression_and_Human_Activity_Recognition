%Tha kanw training kai test kai tha vriskw to pososto tou classification an
%einai swsto tha to vazw se mia metabliti thn opoia tha diairw me to sinolo
var1=0;var2=0;
for i_r=1:200
    [AlphaY, SVs, Bias, Parameters, nSV, nLabel]=SVMTrain(CV.LOO_train{i_r},label(CV.LOO_label_train{i_r})');
    [ClassRate, DecisionValue, Ns, ConfMatrix, PreLabels]=SVMTest(CV.LOO_test{i_r}, label(CV.LOO_label_test{i_r})', AlphaY, SVs, Bias, Parameters,nSV,nLabel );
    plo=label(CV.LOO_label_test{i_r});
    if PreLabels==1 && label(CV.LOO_label_test{i_r})==1
        var1=var1+1;
    end
    if PreLabels==2 && label(CV.LOO_label_test{i_r})==2
        var2=var2+1;
    end
end

%Gia test me ta dedomena tou training an bgalei 100% eimaste O.K.
% var1=0;var2=0;po1=0;po2=0;po3=0;
% for i_r=1:200
%     [AlphaY, SVs, Bias, Parameters, nSV, nLabel]=SVMTrain(CV.LOO_train{i_r},label(CV.LOO_label_train{i_r})');
%     [ClassRate, DecisionValue, Ns, ConfMatrix, PreLabels]=SVMTest(CV.LOO_train{i_r}, label(CV.LOO_label_train{i_r})', AlphaY, SVs, Bias, Parameters,nSV,nLabel );
%     var1=size(find(PreLabels~=label(CV.LOO_label_train{1})'),2)>0;
%     if var1>0
%     po1=po1+var1;
%     po3=i_r;
%     end    
%     if var2>0    
%     var2=size(find(PreLabels==label(CV.LOO_label_train{1})'),2) >0;
%     po2=po2+var2;
%     end
% end
