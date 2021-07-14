classdef boa_model
    % Bayesian Optimization of an ml_model or a cnn_model
    
    properties
        model_obj
        filename
        pth
        parameter_options
        boa_options
        result
        train_trace
        val_trace
        test_trace
    end
    
    methods
         function boamdl = boa_model(model_obj, exp_name, pth)
            
            boamdl.model_obj = model_obj;        
            boamdl.result = [];
            boamdl.train_trace = [];
            boamdl.val_trace = [];
            boamdl.test_trace = [];
            
            if nargin < 3 || isempty(pth); pth = cd; end
            if nargin < 2 || isempty(exp_name)
                time = clock; 
                exp_name = strcat('BOA-Model_',num2str(time(1)),'-',num2str(time(2)),...
                    '-',num2str(time(3)),'-',num2str(time(4)),'-',num2str(time(5)));
            end

            boamdl.filename = exp_name;

            cd(pth)
            mkdir(exp_name)
            cd(exp_name)
            boamdl.pth = cd;
            boamdl.model_obj.pth = cd;
            save(exp_name,'boamdl');
            addpath(strcat(pth,'\',exp_name));
         end
         
         function self = optimize(self, model_num, boa_options, parameter)
            warning('off','all');
             
             if nargin < 3 && isempty(model_num) || nargin < 2
                model_num = size(self.boa_options,1);
            end
            if nargin > 2 && isempty(model_num)
                i = size(self.boa_options,1);
                self.boa_options{i+1,1} = boa_opt(boa_options);
            elseif nargin > 2 && ~(isempty(model_num))
                self.boa_options{model_num,1} = boa_opt(boa_options);         
            end
            if isempty(model_num)
                model_num = size(self.boa_options,1);
            end
            if nargin < 4 || isempty(parameter)
                error('You have to specifie parameter to optimize!');
            end
            boa_opts = self.boa_options{model_num,1};       

            % create experiment folder
            cd(self.pth)
            mkdir(strcat('Versuch_', num2str(model_num)));
            cd(strcat('Versuch_', num2str(model_num)))
            self.model_obj.pth = cd;
            
            
            [params, default_parameter] = defineparameter(parameter);
            names =fieldnames(params);
            for i=1:length(names)
                param_names(1,i)=getfield(params,names{i});
            end

            if strcmp(boa_opts.exit,'Time') == 1
                boa_opts.max_obj_eval = Inf;
            elseif strcmp(boa_opts.exit,'FctEval') == 1
                boa_opts.max_time = Inf;
            end
            
            fitfunhandel = @(params)fitfun(params, default_parameter, self.model_obj, boa_opts);
            
            result = bayesopt(fitfunhandel,param_names,...
                'AcquisitionFunctionName',boa_opts.aquisition_fct,...
                'Verbose',boa_opts.verbose,'ExplorationRatio',boa_opts.exp_ratio,...
                'NumSeedPoints',boa_opts.num_seed,...
                'ConditionalVariableFcn',@(parameter)condvariablefcn(parameter),...
                'XConstraintFcn',@(parameter)xconstraint(parameter,default_parameter),...
                'IsObjectiveDeterministic',false,'MaxTime',boa_opts.max_time,...
                'MaxObjectiveEvaluations',boa_opts.max_obj_eval,'Kernel',boa_opts.kernel); % 'OutputFcn',@outputfun,
            
            r.totalTime = result.TotalElapsedTime;
            r.minObj = result.MinObjective;
            r.estminObj = result.MinEstimatedObjective;
            r.xminObj = result.XAtMinObjective;
            r.xestminObj = result.XAtMinEstimatedObjective;
            r.xTrace = result.XTrace;
            r.ObjTrace = result.ObjectiveTrace;
            
            parameter.partition{2}=[];
            parameter.nrepeats = 1;
            [temp, ~, r.testResult] = fitfun(result.XAtMinObjective, parameter, self.model_obj, boa_opts);
            r.testObj = 1-temp;
            
            self.result{model_num} = r;

            self = trace_calculater(self, result, parameter, boa_opts, model_num);
            
            cd(self.pth);
            boamdl = self;
            save(self.filename,'boamdl');
            warning('on','all');
        end
    end
end

%--------------------------------------------------------------------------
%supporting function
function [param, parameter] = defineparameter(parameter)

name = fieldnames(parameter);

for i = 1: length(name)
    if iscell(parameter.(name{i})) && size(parameter.(name{i}),1) > 1 && isstr(parameter.(name{i}){1})
        opt = 1;
        type = 'categorical';
        transform = 'none';
    elseif iscell(parameter.(name{i})) && size(parameter.(name{i}),1) > 2 && ~isstr(parameter.(name{i}){1})
        opt = 1;
        type = 'categorical';
        transform = 'none'; 
        for n = 1:length(parameter.(name{i}))
            parameter.(name{i}){n} = num2str(parameter.(name{i}){n});
        end
    elseif iscell(parameter.(name{i})) && size(parameter.(name{i}),1) == 2 && ~isstr(parameter.(name{i}){1})
        opt = 1;
        if mod(parameter.(name{i}){1},1) == 0 && mod(parameter.(name{i}){2},1) == 0
            type = 'integer';
        elseif mod(parameter.(name{i}){1},1) ~= 0 || mod(parameter.(name{i}){2},1) ~= 0
            type = 'real';
        end
        if ~isinf(abs(log10(parameter.(name{i}){1})-log10(parameter.(name{i}){2}))) && abs(log10(parameter.(name{i}){1})-log10(parameter.(name{i}){2})) >= 2
            transform = 'log';
        else
            transform = 'none';
        end
    else
        opt = 0;
    end
    
    if opt == 1 && strcmp(type, 'categorical')
       param.(name{i}) = optimizableVariable(name{i},parameter.(name{i}),'Type',type,'Optimize',true,'Transform',transform);
       parameter = rmfield(parameter,name{i});
    elseif opt == 1 && ~strcmp(type, 'categorical')
       param.(name{i}) = optimizableVariable(name{i},[parameter.(name{i}){1:end}],'Type',type,'Optimize',true,'Transform',transform);
       parameter = rmfield(parameter,name{i});
    end
end

end

function [objectiv, constraints, user_data] = fitfun(params, parameter, model, options)
constraints = [];    
params = table2struct(params);
names = fieldnames(params);
for i = 1:length(names)
    var = (params.(names{i}));
    if iscategorical(var)
        var = char(var);
    end
    var_ = str2double(var);
    if ~isnan(var_)
        parameter.(names{i}) = var_;
    else
        parameter.(names{i}) = var;
    end
end

parameter = parameter_correcter(parameter);

if exist(strcat(model.pth,'\',model.filename,'.mat')) ~= 0
    model = load(strcat(model.pth,'\',model.filename));
    name = fieldnames(model);
    model = model.(name{1});
end

model = model.train_model([], parameter);

user_data = model.results(end);

if strcmp(parameter.mode, 'classification')
    if ~isempty(model.results(end).validation)
        if strcmp(options.metric, 'kappa') 
            objectiv = 1 - mean([model.results(end).validation.cohenKappa]);
        else
            objectiv = 1 - mean([model.results(end).validation.randAccuracy]);
        end
    else
        if strcmp(options.metric, 'kappa') 
            objectiv = 1 - mean([model.results(end).test.cohenKappa]);
        else
            objectiv = 1 - mean([model.results(end).test.randAccuracy]);
        end        
    end
elseif strcmp(parameter.mode, 'regression')
    if ~isempty(model.results(end).r2_val)
        if strcmp(options.metric, 'r2')
            objectiv = 1 - mean([model.results(end).r2_val]);
        else
            objectiv = mean([model.results(end).RMSE_val]);
        end
    else
        if strcmp(options.metric, 'r2')
            objectiv = 1 - mean([model.results(end).r2_test]);
        else
            objectiv = mean([model.results(end).RMSE_test]);
        end
    end
 end
end

function parameternew = condvariablefcn(parameter)

%global modeltype
parameternew = parameter;

% noise constraints
if ismember('sg_degree',parameter.Properties.VariableNames)==1;parameternew.sg_degree(parameternew.noise ~= 'savitzky-golay') = NaN;end
if ismember('sg_points',parameter.Properties.VariableNames)==1;parameternew.sg_points(parameternew.noise ~= 'savitzky-golay') = NaN;end
if ismember('sg_deriv',parameter.Properties.VariableNames)==1;parameternew.sg_deriv(parameternew.noise ~= 'savitzky-golay') = NaN;end

% algorithm constraints
% disciminant 
if ismember('model',parameter.Properties.VariableNames)==1
    if ismember('da_discrimtyp',parameter.Properties.VariableNames)==1;parameternew.da_discrimtyp(parameternew.model ~= 'da') = '';end
    if ismember('da_delta',parameter.Properties.VariableNames)==1;parameternew.da_delta(parameternew.model ~= 'da') = NaN;end
    if ismember('da_gamma',parameter.Properties.VariableNames)==1;parameternew.da_gamma(parameternew.model ~= 'da') = NaN;end

% knn
    if ismember('knn_numneigh',parameter.Properties.VariableNames)==1;parameternew.knn_numneigh(parameternew.model ~= 'knn') = NaN;end
    if ismember('knn_standardize',parameter.Properties.VariableNames)==1;parameternew.knn_standardize(parameternew.model ~= 'knn') = '';end
    if ismember('knn_distance',parameter.Properties.VariableNames)==1;parameternew.knn_distance(parameternew.model ~= 'knn') = '';end
    if ismember('knn_distanceweight',parameter.Properties.VariableNames)==1;parameternew.knn_distanceweight(parameternew.model ~= 'knn') = '';end
    if ismember('knn_exponent',parameter.Properties.VariableNames)==1;parameternew.knn_exponent(parameternew.model ~= 'knn') = NaN;end

% svm
    if ismember('svm_kernel',parameter.Properties.VariableNames)==1;parameternew.svm_kernel(parameternew.model ~= 'svm') = '';end
    if ismember('svm_standardize',parameter.Properties.VariableNames)==1;parameternew.svm_standardize(parameternew.model ~= 'svm') = '';end
    if ismember('svm_kernelsc',parameter.Properties.VariableNames)==1;parameternew.svm_kernelsc(parameternew.model ~= 'svm') = NaN;end
    if ismember('svm_box',parameter.Properties.VariableNames)==1;parameternew.svm_box(parameternew.model ~= 'svm') = NaN;end
    if ismember('svm_polynom',parameter.Properties.VariableNames)==1;parameternew.svm_polynom(parameternew.model ~= 'svm') = NaN;end


% tree ensemble
    if ismember('ensemble_MaxNumSplits',parameter.Properties.VariableNames)==1;parameternew.ensemble_MaxNumSplits(parameternew.model ~= 'ensemble') = NaN;end
    if ismember('ensemble_MinLeafSize',parameter.Properties.VariableNames)==1;parameternew.ensemble_MinLeafSize(parameternew.model ~= 'ensemble') = NaN;end
    if ismember('ensemble_SplitCriterion',parameter.Properties.VariableNames)==1;parameternew.ensemble_SplitCriterion(parameternew.model ~= 'ensemble') = '';end
    if ismember('ensemble_NumVariablesToSample',parameter.Properties.VariableNames)==1;parameternew.ensemble_NumVariablesToSample(parameternew.model ~= 'ensemble') = NaN;end
    if ismember('ensemble_methode',parameter.Properties.VariableNames)==1;parameternew.ensemble_methode(parameternew.model ~= 'ensemble') = '';end
    if ismember('ensemble_NumLearningCycles',parameter.Properties.VariableNames)==1;parameternew.ensemble_NumLearningCycles(parameternew.model ~= 'ensemble') = NaN;end
    if ismember('ensemble_LearnRate',parameter.Properties.VariableNames)==1;parameternew.ensemble_LearnRate(parameternew.model ~= 'ensemble') = NaN;end

% grnn
    if ismember('grnn_spread',parameter.Properties.VariableNames)==1;parameternew.grnn_spread(parameternew.model ~= 'grnn') = NaN;end

% nn
    if ismember('nn_hiddensize1',parameter.Properties.VariableNames)==1;parameternew.nn_hiddensize1(parameternew.model ~= 'nn') = NaN;end
    if ismember('nn_hiddensize2',parameter.Properties.VariableNames)==1;parameternew.nn_hiddensize2(parameternew.model ~= 'nn') = NaN;end
    if ismember('nn_hiddensize3',parameter.Properties.VariableNames)==1;parameternew.nn_hiddensize3(parameternew.model ~= 'nn') = NaN;end
    if ismember('nn_fitfun',parameter.Properties.VariableNames)==1;parameternew.nn_fitfun(parameternew.model ~= 'nn') = '';end
end
   
if ismember('da_discrimtyp',parameter.Properties.VariableNames)==1
     if ismember('da_gamma',parameter.Properties.VariableNames)==1;parameternew.da_gamma(ismember(parameternew.da_discrimtyp, {'quadratic',...
         'diagQuadratic','pseudoQuadratic'})) = NaN;end
    if ismember('da_delta',parameter.Properties.VariableNames)==1;parameternew.da_delta(ismember(parameternew.da_discrimtyp, {'quadratic',...
        'diagQuadratic','pseudoQuadratic'})) = NaN;   end 
end

    if ismember('svm_polynom',parameter.Properties.VariableNames)==1;parameternew.svm_polynom(parameternew.svm_kernel ~= 'polynomial') = NaN;end
    if ismember('knn_exponent',parameter.Properties.VariableNames)==1;parameternew.knn_exponent(parameternew.knn_distance ~= 'minkowski') = NaN;end
    if ismember('ensemble_LearnRate',parameter.Properties.VariableNames)==1;parameternew.ensemble_LearnRate(parameternew.ensemble_methode == 'Bag') = NaN;end

    if ismember('data',parameter.Properties.VariableNames)==1;parameternew.range(parameternew.data == 'spectra'|parameternew.data == 'meanspectra') = '';end
end

function parameter = parameter_correcter(parameter)
    names = fieldnames(parameter);
    
    if (ismember('xreflection',names) && parameter.xreflection == 1) || (ismember('reflection',names) && parameter.reflection == 1)
        parameter.xreflection = true;
    else
        parameter.xreflection = false;
    end
    if (ismember('yreflection',names) && parameter.yreflection == 1) || (ismember('reflection',names) && parameter.reflection == 1)
        parameter.yreflection = true;
    else
        parameter.yreflection = false;
    end
    if (ismember('translation',names) && parameter.translation ~= 0)
        parameter.xtranslation = [-parameter.translation parameter.translation];
        parameter.ytranslation = [-parameter.translation parameter.translation];
    elseif ismember('xtranslation',names) && ismember('ytranslation',names)
        parameter.xtranslation = [-parameter.xtranslation parameter.xtranslation];
        parameter.ytranslation = [-parameter.ytranslation parameter.ytranslation];        
    end
    if (ismember('scale',names) && parameter.scale ~= 0)
        parameter.xscale = [1-(parameter.scale-1) parameter.scale];
        parameter.yscale = [1-(parameter.scale-1) parameter.scale];
    elseif ismember('xscale',names) && ismember('yscale',names)
        parameter.xscale = [1-(parameter.xscale-1) parameter.xscale];
        parameter.yscale = [1-(parameter.yscale-1) parameter.yscale];        
    end 
    if (ismember('shear',names) && parameter.shear ~= 0)
        parameter.xshear = [-parameter.shear parameter.shear];
        parameter.yshear = [-parameter.shear parameter.shear];
    elseif ismember('xshear',names) && ismember('yshear',names)
        parameter.xshear = [-parameter.xshear parameter.xshear];
        parameter.yshear = [-parameter.yshear parameter.yshear];        
    end 

    if (ismember('ed_sigma',names) && ismember('ed_alpha',names))
        parameter.edtransformation = [parameter.ed_sigma parameter.ed_sigma; parameter.ed_alpha parameter.ed_alpha];
    end
    
    if ismember('gaussnoise',names)
        parameter.gaussnoise = [0 parameter.gaussnoise];
    end    
    
    if ismember('rotation', names) && parameter.rotation == 1
        parameter.rotation = [0 360];
    else
        parameter.rotation = [0 0];
    end
    
    if ismember('layers',names) && ismember('fullyconnect', names)
        parameter.dropout = zeros(1, parameter.layers + parameter.fullyconnect);
    end
    if ismember('dropout_start',names)
        parameter.dropout(1) = parameter.dropout_start;
    end
    if ismember('dropout_last',names)
        parameter.dropout(end) = parameter.dropout_last;
    end
    if ismember('range',names)
        parameter.range = str2num(parameter.range);
    end
end

function boamdl = trace_calculater(boamdl, result, parameter, options, model_num)

data = result.UserDataTrace;

if strcmp(parameter.mode, 'classification')
    if ~strcmp(options.metric, 'kappa') || strcmp(options.metric, 'acc')
        options.metric = 'acc';
    end
        if strcmp(options.metric, 'kappa')
            if ~isempty(data{1}.train)
                for l = 1:size(data,1)
                    boamdl.train_trace(l,model_num) = mean([data{l}.train.cohenKappa]);
                    if l > 1 && boamdl.train_trace(l,model_num) < boamdl.train_trace(l-1,model_num)
                        boamdl.train_trace(l,model_num) = boamdl.train_trace(l-1,model_num);
                    end
                end
            end
            if ~isempty(data{1}.validation)
                for l = 1:size(data,1)
                    boamdl.val_trace(l,model_num) = mean([data{l}.validation.cohenKappa]);
                    if l > 1 && boamdl.val_trace(l,model_num) < boamdl.val_trace(l-1,model_num)
                        boamdl.val_trace(l,model_num) = boamdl.val_trace(l-1,model_num);
                    end
                end
            end            
             if ~isempty(data{1}.test)
                for l = 1:size(data,1)
                    boamdl.test_trace(l,model_num) = mean([data{l}.test.cohenKappa]);
                    if l > 1 && boamdl.test_trace(l,model_num) < boamdl.test_trace(l-1,model_num)
                        boamdl.test_trace(l,model_num) = boamdl.test_trace(l-1,model_num);
                    end
                end
             end
        elseif strcmp(options.metric, 'acc')
            if ~isempty(data{1}.train)
                for l = 1:size(data,1)
                    boamdl.train_trace(l,model_num) = mean([data{l}.train.randAccuracy]);
                    if l > 1 && boamdl.train_trace(l,model_num) < boamdl.train_trace(l-1,model_num)
                        boamdl.train_trace(l,model_num) = boamdl.train_trace(l-1,model_num);
                    end
                end
            end
            if ~isempty(data{1}.validation)
                for l = 1:size(data,1)
                    boamdl.val_trace(l,model_num) = mean([data{l}.validation.randAccuracy]);
                    if l > 1 && boamdl.val_trace(l,model_num) < boamdl.val_trace(l-1,model_num)
                        boamdl.val_trace(l,model_num) = boamdl.val_trace(l-1,model_num);
                    end
                end
            end            
             if ~isempty(data{1}.test)
                for l = 1:size(data,1)
                    boamdl.test_trace(l,model_num) = mean([data{l}.test.randAccuracy]);
                    if l > 1 && boamdl.test_trace(l,model_num) < boamdl.test_trace(l-1,model_num)
                        boamdl.test_trace(l,model_num) = boamdl.test_trace(l-1,model_num);
                    end
                end
             end
        end
    elseif strcmp(parameter.mode, 'regression')
        
        if ~isfield(data{1},'RMSE_test')
            data{1}.RMSE_test = [];
        end
        if ~isfield(data{1},'RMSE_val')
            data{1}.RMSE_val = [];
        end
        if ~isfield(data{1},'r2_test')
            data{1}.r2_test = [];
        end
        if ~isfield(data{1},'r2_val')
            data{1}.r2_val = [];
        end
        
    if ~strcmp(options.metric, 'rmse') || strcmp(options.metric, 'r2')
        options.metric = 'rmse';
    end
    if strcmp(options.metric, 'rmse')
            if ~isempty(data{1}.RMSE_train)
                for l = 1:size(data,1)
                    boamdl.train_trace(l,model_num) = mean([data{l}.RMSE_train]);
                    if l > 1 && boamdl.train_trace(l,model_num) > boamdl.train_trace(l-1,model_num)
                        boamdl.train_trace(l,model_num) = boamdl.train_trace(l-1,model_num);
                    end
                end
            end
            if ~isempty(data{1}.RMSE_val)
                for l = 1:size(data,1)
                    boamdl.val_trace(l,model_num) = mean([data{l}.RMSE_val]);
                    if l > 1 && boamdl.val_trace(l,model_num) > boamdl.val_trace(l-1,model_num)
                        boamdl.val_trace(l,model_num) = boamdl.val_trace(l-1,model_num);
                    end
                end
            end            
             if ~isempty(data{1}.RMSE_test)
                for l = 1:size(data,1)
                    boamdl.test_trace(l,model_num) = mean([data{l}.RMSE_test]);
                    if l > 1 && boamdl.test_trace(l,model_num) > boamdl.test_trace(l-1,model_num)
                        boamdl.test_trace(l,model_num) = boamdl.test_trace(l-1,model_num);
                    end
                end
             end
        elseif strcmp(options.metric, 'r2')
            if ~isempty(data{1}.r2_train)
                for l = 1:size(data,1)
                    boamdl.train_trace(l,model_num) = mean([data{l}.r2_train]);
                    if l > 1 && boamdl.train_trace(l,model_num) < boamdl.train_trace(l-1,model_num)
                        boamdl.train_trace(l,model_num) = boamdl.train_trace(l-1,model_num);
                    end
                end
            end
            if ~isempty(data{1}.r2_val)
                for l = 1:size(data,1)
                    boamdl.val_trace(l,model_num) = mean([data{l}.r2_val]);
                    if l > 1 && boamdl.val_trace(l,model_num) < boamdl.val_trace(l-1,model_num)
                        boamdl.val_trace(l,model_num) = boamdl.val_trace(l-1,model_num);
                    end
                end
            end            
             if ~isempty(data{1}.r2_test)
                for l = 1:size(data,1)
                    boamdl.test_trace(l,model_num) = mean([data{l}.r2_test]);
                    if l > 1 && boamdl.test_trace(l,model_num) < boamdl.test_trace(l-1,model_num)
                        boamdl.test_trace(l,model_num) = boamdl.test_trace(l-1,model_num);
                    end
                end
             end
         end
    end
end

function tf = xconstraint(parameter,default_parameter)
if ismember('noise',parameter.Properties.VariableNames)
    tf3 = parameter.noise == 'none';
else
    tf3 = logical(ones(height(parameter),1));
end

if ismember('sg_points',parameter.Properties.VariableNames) && ismember('sg_degree',parameter.Properties.VariableNames)    
    tf1 = mod(parameter.sg_points,2) == 1;
    tf2 = parameter.sg_points >= parameter.sg_degree+1;
    tf = tf1 & tf2;
elseif ismember('sg_points',parameter.Properties.VariableNames) && ~ismember('sg_degree',parameter.Properties.VariableNames)   
    tf1 = mod(parameter.sg_points,2) == 1;
    tf2 = parameter.sg_points >= default_parameter.sg_degree(1)+1;
    tf = tf1 & tf2;
elseif ~ismember('sg_points',parameter.Properties.VariableNames) && ismember('sg_degree',parameter.Properties.VariableNames)
    tf = parameter.sg_degree <= default_parameter.sg_points(1);
else
    tf = ones(height(parameter),1)==1;
end

if ismember('sg_deriv',parameter.Properties.VariableNames) && ismember('sg_degree',parameter.Properties.VariableNames)
    tf1 = parameter.sg_deriv <= parameter.sg_degree;
elseif ~ismember('sg_deriv',parameter.Properties.VariableNames) && ismember('sg_degree',parameter.Properties.VariableNames)
    tf1 = default_parameter.sg_deriv(1) <= parameter.sg_degree;
elseif ismember('sg_deriv',parameter.Properties.VariableNames) && ~ismember('sg_degree',parameter.Properties.VariableNames)
    tf1 = parameter.sg_deriv <= default_parameter.sg_degree(1);
else
    tf1 = logical(ones(height(parameter),1));
end

tf = tf & tf1;
tf = tf | tf3;

end




