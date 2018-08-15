% creates the matrixes of the wieghts
W1=zeros(200,1024);
W2=zeros(200,200);
W3=zeros(200,200);
W4=zeros(200,200);
W5=zeros(200,200);
W6=zeros(4,200);

% line below is in comments as its only there, if you wish to continue training with the same wieghts
%load('CountBlobs01.mat');

Alpha = 0.002;


% Teaching the system using the training data

% reads the crrect file 
FileName='TrainingData.txt' ;


% loop for reapeat the learning process
for k=[1:180000];
 % reads the training data and extract the info 
  M = dlmread(FileName,' ',[k-1,0,k-1,1024]);
  
  N = M(1);
  
  % PictureVector is the "input" of the NN
  PictureVector = M([2:1025])'; 
  
  % answerVector is the "output" of the NN
  %line below is seting the output to be a one where the number chosen is 
  AnswerVector = zeros(4,1) ;
  if N > 0
    AnswerVector(N) = 1;
  end
  
  [deltaW1,deltaW2,deltaW3,deltaW4,deltaW5,deltaW6] = BackPropagationLearn(W1,W2,W3,W4,W5,W6,PictureVector,AnswerVector,Alpha); 
  
  % changes the matrixes by there deltas respectivly
  W1 = W1 + deltaW1;
  W2 = W2 + deltaW2;
  W3 = W3 + deltaW3;
  W4 = W4 + deltaW4;
  W5 = W5 + deltaW5;
  W6 = W6 + deltaW6;
  
  %displays the deltas of each layer every 1000 times
  if k==floor(k/1000)*1000  
    fprintf('k=%i, dW : %5f %5f %5f %5f %5f %5f \n' , k, sum(sum(abs(deltaW1))), sum(sum(abs(deltaW2))), sum(sum(abs(deltaW3))),   sum(sum(abs(deltaW4))),   sum(sum(abs(deltaW5))),   sum(sum(abs(deltaW6))));
    fflush(stdout);
  end
  
end

% saves the matrixes so that I can use them later
save('CountBlobs02.mat','W1','W2','W3','W4','W5','W6');  

