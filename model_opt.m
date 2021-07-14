classdef model_opt

properties 
    model = 'ensemble';                  % 'ensemble', 'knn', 'da', 'svm'
    mode = 'regression';                 % 'classification', 'regression'
    save_model = 0;
    
    noise = 'none';
    sg_degree = 2;
    sg_points = 3;
    sg_deriv = 0;
    scaling = 'none';
    add_data = 0;
    
	da_discrimtyp = 'linear';            %'linear' 'quadratic' 'diagQuadratic''pseudoLinear' 'diagLinear' 'pseudoQuadratic'
    da_delta = 0;
    da_gamma = 1;
    
    knn_numneigh = 3;
    knn_standardize = 'off';
    knn_distance = 'euclidean';           % 'cityblock' 'chebychev' 'correlation' 'cosine' 'euclidean' 'hamming' 'jaccard' 'mahalanobis' 'minkowski' 'seuclidean' 'spearman'
    knn_distanceweight = 'equal';         % 'equal' 'inverse' 'squaredinverse'	
    knn_exponent = 1;
    
    svm_polynom = 3;
    svm_box = 1;
    svm_kernelsc  = 1;
    svm_standardize = 'off';
    svm_kernel = 'gaussian';              % 'gaussian' 'linear' 'polynomial'
    
    ensemble_MaxNumSplits = 100;
    ensemble_MinLeafSize = 1;
    ensemble_SplitCriterion ='gdi';       % 'gdi' 'deviance' 'twoing'
    ensemble_NumVariablesToSample = 1;
    ensemble_methode = 'AdaBoost';    	  % 'Bag' 'RUSBoost' 'AdaBoost'
    ensemble_NumLearningCycles = 50;
    ensemble_LearnRate = 0.1;

    grnn_spread = 1;

    nn_hiddensize1 = 10;
    nn_hiddensize2 = 0;
    nn_hiddensize3 = 0;
    nn_fitfun = 'trainlm';
end

methods
    function mdlopt = model_opt(options)
        optnames = fieldnames(options);
        for i = 1:length(optnames)
            switch lower(num2str(optnames{i}))
                case 'model'
                    mdlopt.model =  options.(optnames{i});
                case 'mode'
                    mdlopt.mode =  options.(optnames{i});
                case 'savemodel'
                    mdlopt.save_model =  options.(optnames{i});
                case 'noise'
                    mdlopt.noise =  options.(optnames{i});                    
                case 'sg_degree'
                    mdlopt.sg_degree =  options.(optnames{i});
                case 'sg_points'
                    mdlopt.sg_points =  options.(optnames{i});
                case 'sg_deriv'
                    mdlopt.sg_deriv =  options.(optnames{i});
                 case 'scaling'
                    mdlopt.scaling =  options.(optnames{i});
                case 'add_data'
                    mdlopt.add_data =  options.(optnames{i});
                case 'da_discrimtyp'
                    mdlopt.da_discrimtyp =  options.(optnames{i});
                case 'da_delta'
                    mdlopt.da_delta =  options.(optnames{i});
                case 'da_gamma'
                    mdlopt.da_gamma =  options.(optnames{i});
                    
                case 'knn_numneigh'
                    mdlopt.knn_numneigh =  options.(optnames{i});
                case 'knn_standardize'
                    mdlopt.knn_standardize =  options.(optnames{i});
                case 'knn_distance'
                    mdlopt.knn_distance =  options.(optnames{i});
                case 'knn_distanceweight'
                    mdlopt.knn_distanceweight =  options.(optnames{i});
                case 'knn_exponent'
                    mdlopt.knn_exponent =  options.(optnames{i});
                    
                case 'svm_polynom'
                    mdlopt.svm_polynom =  options.(optnames{i});
                case 'svm_box'
                    mdlopt.svm_box =  options.(optnames{i});
                case 'svm_kernelsc'
                    mdlopt.svm_kernelsc =  options.(optnames{i});
                case 'svm_standardize'
                    mdlopt.svm_standardize =  options.(optnames{i});
                case 'svm_kernel'
                    mdlopt.svm_kernel =  options.(optnames{i});
                    
                case 'ensemble_maxnumsplits'
                    mdlopt.ensemble_MaxNumSplits =  options.(optnames{i});
                case 'ensemble_minleafsize'
                    mdlopt.ensemble_MinLeafSize =  options.(optnames{i});
                case 'ensemble_splitcriterion'
                    mdlopt.ensemble_SplitCriterion =  options.(optnames{i});
                case 'ensemble_numvariablestosample'
                    mdlopt.ensemble_NumVariablesToSample =  options.(optnames{i});
                case 'ensemble_methode'
                    mdlopt.ensemble_methode =  options.(optnames{i});
                case 'ensemble_numlearningcycles'
                    mdlopt.ensemble_NumLearningCycles =  options.(optnames{i});
                case 'ensemble_learnrate'
                    mdlopt.ensemble_LearnRate =  options.(optnames{i});
                                        
                case 'grnn_spread'
                    mdlopt.grnn_spread =  options.(optnames{i});
                    
                case 'nn_hiddensize1'
                    mdlopt.nn_hiddensize1 =  options.(optnames{i});
                case 'nn_hiddensize2'
                    mdlopt.nn_hiddensize2 =  options.(optnames{i});
                case 'nn_hiddensize3'
                    mdlopt.nn_hiddensize3 =  options.(optnames{i});
                case 'nn_fitfun'
                    mdlopt.nn_fitfun =  options.(optnames{i});
                otherwise
            end
        end
    end
end
end