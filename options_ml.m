function options = options_ml()

%model
    options.model = 'grnn';                     % 'ensemble', 'knn', 'da', 'svm' 'grnn' 'nn'
    options.mode = 'regression';

% pre-processing
    options.noise = 'none';   %'none', 'savitzky-golay'
    options.sg_degree = 1;
    options.sg_points = 7;
    options.sg_deriv = 0;
    options.scaling = '1'; %'none' '1' '2' 'inf' 'SNV' 'minmax'
    options.add_data = 0;
    
% model options
% knn
    options.knn_numneigh = 3;
    options.knn_standardize = false;
    options.knn_distance = 'euclidean';             % 'cityblock' 'chebychev' 'correlation' 'cosine' 'euclidean' 'hamming' 'jaccard' 'mahalanobis' 'minkowski' 'seuclidean' 'spearman'
    options.knn_distanceweight = 'equal';           % 'equal' 'inverse' 'squaredinverse'	
    options.knn_exponent = 2;
    
% ensemble
    options.ensemble_MaxNumSplits = 5;
    options.ensemble_MinLeafSize = 44;
    options.ensemble_SplitCriterion ='gdi';         % 'gdi' 'deviance' 'twoing'
    options.ensemble_NumVariablesToSample = [];
    options.ensemble_methode = 'AdaBoost';          % 'Bag' 'RUSBoost' 'AdaBoost'
    options.ensemble_NumLearningCycles = 143;
    options.ensemble_LearnRate = 0.33333;
    
% da
	options.da_discrimtyp = 'linear';               %'linear' 'quadratic' 'diagQuadratic''pseudoLinear' 'diagLinear' 'pseudoQuadratic'
    options.da_delta = 0;
    options.da_gamma = 0;
    
% svm
    options.svm_polynom = 3;
    options.svm_box = 1;
    options.svm_kernelsc  = 1;
    options.svm_standardize = false;
    options.svm_kernel = 'linear';                % 'gaussian' 'linear' 'polynomial'

% grnn
    options.grnn_spread = 1;

% nn
    options.nn_hiddensize1 = 30;
    options.nn_hiddensize2 = 10;
    options.nn_hiddensize3 = 0;
    options.nn_fitfun = 'trainlm';
    
% training options
    options.validationmode = 'partition';           % 'cv', 'partition'
    options.folds = 2;
    options.partition = {0.5 0.25 0.25};
    options.fitnessfunction = 'kappa';           % 'kappa', 'accuracy'
    options.nrepeats = 1;
    options.data = 'scores';
    options.range = 1:10;
    options.data_partition = 1;
    options.zscore = 1;
    options.savemodel = 1;
end