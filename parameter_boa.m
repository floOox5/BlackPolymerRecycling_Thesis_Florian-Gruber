function parameter = parameter_boa()

%--------------------------------------------------------------------------
% cnn
% parameter.Normalization = 'none';%c
% parameter.ImageData = 'scores';%c
% parameter.DataRange =1:3;
% parameter.ImageSize = [48 48];
% parameter.ResizeMethod = 'resize';
% 
% parameter.BackgroundExcecution = true;
% parameter.xreflection = 0;
% parameter.yreflection = 0;
% parameter.reflection = {0; 1};
% parameter.xtranslation = 0;
% parameter.ytranslation = 0;
% parameter.translation = {0; 16};
% parameter.xscale = 1;
% parameter.yscale = 1;
% parameter.scale = {1; 1.25};
% parameter.xshear = 0;
% parameter.yshear = 0;
% parameter.shear = {0; 25};
% parameter.ed_sigma = 0;
% parameter.ed_alpha = 0;
% parameter.gaussnoise = {0; 0.2};%
% parameter.rotation = {0; 1};
% 
% parameter.mode = 'classification';
% parameter.net = [];
% parameter.safeNet = 0;
% parameter.layers = {2;4};%i
% parameter.depth = {1;3};%i
% parameter.filtermode = 'double';
% parameter.filter = {8; 48};%i
% parameter.batchnormalization = 1;%c
% parameter.relu = 'relu';%c
% parameter.fullyconnect = {1; 3};%i
% parameter.sizefullyconnect = {'25'; '50'; '75'; '100'; '150'; '200'};%i
% parameter.convolutionsize = {1;3;5;7};%i
% parameter.convolutionstride = 1;%i
% parameter.poolingmode = {'average';'max'};%c
% parameter.poolingsize = [2 2];
% parameter.poolingstride = [2 2];
% parameter.dropout_start = 0;
% parameter.dropout_end = {0; 0.5};
% 
% parameter.momentum = {0.5; 0.999};
% parameter.learnrate = {0.001; 1};
% parameter.learnrateschedule = 'none';
% parameter.learnratedropperiod = 100;
% parameter.learnratedropfactor = 0.1;
% parameter.l2regularization = {0.00001; 1};
% parameter.shuffle = 'every-epoch';
% parameter.batchsize = 128;
% parameter.plots = 'none';
% parameter.verbose = 0;
% parameter.validationfrequency = 3;
% parameter.validationpatience = 3;
% parameter.maxepochs = 100;
% parameter.datasplit = {0.5, 0.25, 0.25};
% parameter.repeats = 3;

%--------------------------------------------------------------------------
% ml_model
%model
    parameter.model = {'svm'};                     % 'ensemble', 'knn', 'da', 'svm', 'grnn', 'nn'
    parameter.mode = 'regression';
    
% pre-processing
    parameter.noise = {'none'; 'savitzky-golay'};         %{'none'; 'savitzky-golay'};
    parameter.sg_degree = {1; 5};             %{1; 5};
    parameter.sg_points = {3;21};            %{3;21};
    parameter.sg_deriv = 0;
    parameter.scaling = {'none'; '1'; '2'; 'inf'; 'SNV'; 'minmax'};               
    parameter.add_data = 0;
    
%model options
%knn
%     parameter.knn_numneigh = {1;11};
%     parameter.knn_standardize = false;
%     parameter.knn_distance = {'cityblock'; 'chebychev'; 'correlation'; 'cosine'; 'euclidean'; 'hamming'; 'jaccard'; 'mahalanobis'; 'minkowski'; 'seuclidean'; 'spearman'};             % 'cityblock' 'chebychev' 'correlation' 'cosine' 'euclidean' 'hamming' 'jaccard' 'mahalanobis' 'minkowski' 'seuclidean' 'spearman'
%     parameter.knn_distanceweight = {'equal'; 'inverse'; 'squaredinverse'};           % 'equal' 'inverse' 'squaredinverse'	
%     parameter.knn_exponent = {0.5;3};
    
%ensemble
%     parameter.ensemble_MaxNumSplits = {1;101};
%     parameter.ensemble_MinLeafSize = {1; 101};
%     parameter.ensemble_SplitCriterion ='gdi';         % 'gdi' 'deviance' 'twoing'
%     parameter.ensemble_NumVariablesToSample = 'all';
%     parameter.ensemble_methode = {'Bag'; 'LSBoost'};          % 'Bag' 'RUSBoost' 'AdaBoost'
%     parameter.ensemble_NumLearningCycles = {10; 500};
%     parameter.ensemble_LearnRate = {0.001;1};
    
%da
% 	parameter.da_discrimtyp = 'linear';               %'linear' 'quadratic' 'diagQuadratic''pseudoLinear' 'diagLinear' 'pseudoQuadratic'
%     parameter.da_delta = {0.00000000000001;1};
%     parameter.da_gamma = {0.00000000000001;1};
    
%svm
    parameter.svm_polynom = {2;4};
    parameter.svm_box = {0.001;1000};
    parameter.svm_kernelsc  = {0.001;1000};
    parameter.svm_standardize = false;
    parameter.svm_kernel = {'gaussian'; 'linear'; 'polynomial'};                % 'gaussian' 'linear' 'polynomial'

%grnn
%   parameter.grnn_spread = {0.00001; 10000};
% 
% %nn
%     parameter.nn_hiddensize1 = {2; 5};
%     parameter.nn_hiddensize2 = {0; 5};
%     parameter.nn_hiddensize3 = [0];
%     parameter.nn_fitfun = {'trainlm';'trainbr';'trainscg';'trainrp'};
    
%training options
    parameter.validationmode = 'cv';           % 'cv', 'partition'
    parameter.folds = 10;
    parameter.partition = {0.5 0 0.5};
    parameter.fitnessfunction = 'rmse';           % 'kappa', 'accuracy'
    parameter.nrepeats = 1;
    parameter.data = 'scores';
    parameter.range = {1:2;1:3;1:4;1:5};
    parameter.data_partition = 1;
    parameter.zscore = 1;
    
end