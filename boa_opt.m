classdef boa_opt

properties
    
    exit = 'FctEval'; % 'FctEval' 'Fitness' 'Time'
	aquisition_fct = 'lower-confidence-bound'; %'expected-improvement-per-second-plus' (default) | 'expected-improvement' | 'expected-improvement-plus' | 'expected-improvement-per-second' | 'lower-confidence-bound' | 'probability-of-improvement'
	verbose = 1;
	exp_ratio = 0.5;
	max_obj_eval = 30;
	num_seed = 4;
	max_time = 72000;
	%fitness = 0.05;
	kernel = 'ardmatern32';
    metric = 'acc';
end

methods
    function boaopt = boa_opt(options)
        optnames = fieldnames(options);
        for i = 1:length(optnames)
            switch lower(num2str(optnames{i}))
                case 'exit'
                    boaopt.exit = options.(optnames{i});
                case 'aquisitionfunction'
                    boaopt.aquisition_fct = options.(optnames{i});
                case 'verbose'
                    boaopt.verbose = options.(optnames{i});
                case 'explorationratio'
                    boaopt.exp_ratio = options.(optnames{i});
                case 'maxobjectiveevaluations'
                    boaopt.max_obj_eval = options.(optnames{i});
                case 'numseedpoints'
                    boaopt.num_seed = options.(optnames{i});
                case 'maxtime'
                    boaopt.max_time = options.(optnames{i});
                %case 'fitness'
                %    boaopt.fitness = options.(optnames{i});
                case 'kernel'
                    boaopt.kernel = options.(optnames{i});
                case 'metric'
                    boaopt.metric = options.(optnames{i});
                otherwise

            end
        end
    end
end
end