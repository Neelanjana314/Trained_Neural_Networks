% Translates Matlab networks(as LeakyReluLayer) to NNV FFNN networks
function nnvnet = LeakyRelu_Matlabnet_2_NNV_FFNN(net,std)
    n = length(net.Layers)/2-1;
    for i=1:n-1
        Layers(i)=LayerS(double(net.Layers(i*2,1).Weights),double(net.Layers(i*2,1).Bias),'lrelu');
    end
    Layers(n)=LayerS(double(net.Layers(2*n,1).Weights),double(net.Layers(2*n,1).Bias),'softmax');
    nnvnet=FFNNS(Layers,'lrelunet');
    
    %test FFNN networks
    load X_Test.mat
    load X_Test_4D.mat
    load Y_Test_cat.mat
    Y_Test = double(YYtest);
    Y_Pred=net.classify(XXtest);
    Y_Pred=double(Y_Pred);
    mean = 33.3184;
    for i=1:10000
    Y_nnvnet(i)=classify(nnvnet,(X_Test(i,:)'-mean)/std);
    end
    fprintf('test_accuracy of Matlab network %f percent \n',sum(Y_Pred==Y_Test)/10000*100);
    fprintf('test_accuracy of FFNN network %f percent \n',sum(Y_nnvnet'==Y_Test)/10000*100);
    fprintf('comparision of Matlab network and FFNN network %f percent \n',sum(Y_nnvnet'==Y_Pred)/10000*100);
end
