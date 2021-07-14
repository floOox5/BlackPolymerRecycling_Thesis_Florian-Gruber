classdef train_opt_ml

properties
    data = 'meanscores';             % 'scores', 'data'
    range = 1:5;
    data_partition = 1;
    validation = 'partition';   %'cv' 'partition'
    nkfold = 5;
    partition = [0.5 0.2 0.3];
    fitness = 'kappa';          % 'kappa', 'acc'
    nrepeats = 1;
    zscore = 0;
     
end

methods
    function trainopt = train_opt_ml(options)
        optnames = fieldnames(options);
        for i = 1:length(optnames)
            switch lower(num2str(optnames{i}))
                case 'validationmode'
                    trainopt.validation =  options.(optnames{i});
                case 'folds'
                    trainopt.nkfold =  options.(optnames{i});
                case 'partition'
                    trainopt.partition =  options.(optnames{i});
                case 'fitnessfunction'
                    trainopt.fitness =  options.(optnames{i});
                case 'nrepeats'
                    trainopt.nrepeats =  options.(optnames{i});
                case 'data'
                    trainopt.data =  options.(optnames{i});
                case 'range'
                    trainopt.range =  options.(optnames{i});
                case 'data_partition'
                    trainopt.data_partition =  options.(optnames{i});
                case 'zscore'
                    trainopt.zscore =  options.(optnames{i});
                otherwise
            end
        end
    end
end
end