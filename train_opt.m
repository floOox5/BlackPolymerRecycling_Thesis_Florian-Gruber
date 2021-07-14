classdef train_opt

properties
    momentum = 0.9;
    lr = 0.001;
    lr_schedule = 'none';
    lr_drop_period = 1;
    lr_drop_factor = 1;
    l2_reg = 0.0005;
    shuffle = 'every-epoch';
    bs = 128;
    plots = 'training-progress'
    verbose = 0;
    valfreq = 2;
    valpatience = 5;
    maxepochs = 100;
    data_split = [0.7 0.15 0.15];
    num_repeats = 1;
     
end

methods
    function trainopt = train_opt(options)
        optnames = fieldnames(options);
        for i = 1:length(optnames)
            switch lower(num2str(optnames{i}))
                case 'momentum'
                    trainopt.momentum =  options.(optnames{i});
                case 'learnrate'
                    trainopt.lr =  options.(optnames{i});
                case 'learnrateschedule' 
                    trainopt.lr_schedule =  options.(optnames{i});
                case 'learnratedropperiod'            
                    trainopt.lr_drop_period =  options.(optnames{i});
                case 'learnratedropfactor'            
                    trainopt.lr_drop_factor =  options.(optnames{i});
                case 'l2regularization'                
                    trainopt.l2_reg = options.(optnames{i});
                case 'shuffle'                    
                    trainopt.shuffle =  options.(optnames{i});
                case 'batchsize'
                    trainopt.bs =  options.(optnames{i});
                case 'plots'                    
                    trainopt.plots =  options.(optnames{i});
                case 'validationfrequency'             
                    trainopt.valfreq =  options.(optnames{i});
                case 'validationpatience'              
                    trainopt.valpatience = options.(optnames{i});
                case 'verbose'                    
                    trainopt.verbose =  options.(optnames{i});
                case 'maxepochs'                    
                    trainopt.maxepochs =  options.(optnames{i});
                case 'datasplit'                    
                    trainopt.data_split =  options.(optnames{i});
                case 'repeats'
                    trainopt.num_repeats =  options.(optnames{i});
                otherwise

            end
        end
    end
end
end