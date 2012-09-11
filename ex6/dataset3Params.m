function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
%C = 1;
%sigma = 0.3;


%Some inits
C = 0.01;

err = 1000;
optsigma=0;
optC=0;

while C < 32
    sigma = 0.01;
    while sigma < 32
        model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        predictions = svmPredict(model, Xval);
        calerr = mean(double(predictions ~= yval));

        if err > calerr
            err = calerr
            optsigma = sigma
            optC = C
        end

        sigma *= 3
    
    end
    C *= 3
end

C=optC;
sigma=optsigma;
        

end
