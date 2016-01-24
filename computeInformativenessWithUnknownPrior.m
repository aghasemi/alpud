function informativeness = computeInformativenessWithUnknownPrior( positiveData,unlabeledData,querySamples )
%COMPUTEINFORMATIVENESS Computes the amount of informativeness of a data
%sample in a semi-supervised one-class learning scenration where only
%positive and unlabeled data are avaibale. Please see the ALPUD paper
%(cited in the README) for more information. This function does not requires that a
%prior knowledge of the proportion of positive examples in the sample space
%(P) be available. Instead it computes the expected value over all possible
%Ps using a uniform prior distribution. 
%  positiveData: 
%    Data samples known to belong to the target class
%  unalabeledData:
%   Data samples we don't know whether they belong to target class or not
%  querySamples
%   Data samples for which we want to compute the informativeness


%% p(x|+)
positiveLikelihood=ksdensity(positiveData,querySamples);

%% p(x)
totalLikelihood=ksdensity([positiveData;unlabeledData],querySamples);

%% a_x=p(x|+)/p(x)
aOfX=positiveLikelihood./totalLikelihood;

%% Now we can compute the informativeness based on Eq. 8 in the paper
informativeness=(1-aOfX).*sgn(.5-aOfX);
end

