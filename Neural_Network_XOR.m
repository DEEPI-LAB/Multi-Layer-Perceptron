% *********************************************
% Multi-Layer-Perceptron - Nueral Networks
% Jongwon Kim
% jw.kim@sch.ac.kr
% activation function = sigmoid
% *********************************************

clear all
%% Training Set (XOR Problem)
x = [0 0 ; 0 1 ; 1 0 ; 1 1];
y = [0.0000 1.0000 1.0000 0.0000];

b1 = zeros(1,2);
b2 = zeros;
w1 = zeros(2,2);
w2 = zeros(2,1);

epoch = 20000;
deltha = 1;

%% Neural Networks 2-2-1 
for i=1:epoch

   % x = flipud(x);
   % y = flipud(y);
    
    for j=1:length(x)
        
        %feedforward
        Z =  1 ./ (1 + exp(- (x(j,:)*w1+b1))); 
        Y =  1 ./ (1 + exp(- (Z*w2+b2)));        
        
        %error
        E(j) = y(j) - Y;
        
        %backpropagation
        alpha2 = E(j)*(Y)*(1-Y) ;
        alpha1 = alpha2*((Z).*(1-Z)).*w2';
        
        % update
        w2 = w2 + (deltha*alpha2*Z)' ;
        b2 = b2 + deltha*alpha2 ;
  
        w1 = w1 + deltha*alpha1'*x(j,:);
        b1 = b1 + deltha*alpha1;
       
        result(j) = Y;
  
    end  
    %mean square error
    mse(i) = mean(E.^2);

   txt1 = ['Target Value             ' sprintf('%.4f   ',y)];
   txt2= ['Predicted Value        ' sprintf('%.4f   ',result)];
   txt3= ['Epochs   ' sprintf('%d',i)];
   disp(txt1);
   disp(txt2);
   disp(txt3);
   clc
end
%mean square error
plot(mse);
    
