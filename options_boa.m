function options = options_boa()

options.Exit = 'FctEval';
options.AquisitionFunction = 'probability-of-improvement'; %'expected-improvement-per-second-plus' (default) | 'expected-improvement' | 'expected-improvement-plus' | 'expected-improvement-per-second' | 'lower-confidence-bound' | 'probability-of-improvement'
options.Verbose = 1;
options.ExplorationRatio = 0.5;
options.MaxObjectiveEvaluations = 50;
options.NumSeedPoints = 4;
options.MaxTime = 72000;
options.Kernel = 'ardmatern32';
options.Metric = 'rmse'; % 'acc', 'rmse' 'r2'

end