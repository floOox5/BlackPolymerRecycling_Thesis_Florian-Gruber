classdef random_model
    % Bayesian Optimization of an ml_model or a cnn_model
    
    properties
        model_obj
        filename
        pth
        parameter_options
        random_options
        result
        train_trace
        val_trace
        test_trace
    end
    
    methods
         function randmdl = random_model(model_obj, exp_name, pth)
            
            randmdl.model_obj = model_obj;        
            randmdl.result = [];
            randmdl.train_trace = [];
            randmdl.val_trace = [];
            randmdl.test_trace = [];
            
            if nargin < 3 || isempty(pth); pth = cd; end
            if nargin < 2 || isempty(exp_name)
                time = clock; 
                exp_name = strcat('RandomOpt-Model_',num2str(time(1)),'-',num2str(time(2)),...
                    '-',num2str(time(3)),'-',num2str(time(4)),'-',num2str(time(5)));
            end

            randmdl.filename = exp_name;

            cd(pth)
            mkdir(exp_name)
            cd(exp_name)
            randmdl.pth = cd;
            randmdl.model_obj.pth = cd;
            save(exp_name,'randmdl');
            addpath(strcat(pth,'\',exp_name));
         end
         
         function self = optimize(self, model_num, random_options, parameter)
            warning('off','all');
             
             if nargin < 3 && isempty(model_num) || nargin < 2
                model_num = size(self.random_options,1);
            end
            if nargin > 2 && isempty(model_num)
                i = size(self.random_options,1);
                self.random_options{i+1,1} = rand_opt(random_options);
            elseif nargin > 2 && ~(isempty(model_num))
                self.random_options{model_num,1} = rand_opt(random_options);         
            end
            if isempty(model_num)
                model_num = size(self.random_options,1);
            end
            if nargin < 4 || isempty(parameter)
                error('You have to specifie parameter to optimize!');
            end
            rand_opts = self.random_options{model_num,1};       

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

            if strcmp(rand_opts.exit,'Time') == 1
                rand_opts.max_obj_eval = Inf;
            elseif strcmp(rand_opts.exit,'FctEval') == 1
                rand_opts.max_time = Inf;
            end

            i = 1;
            b = 0;
            fprintf('| Nr.   | last Fitness | last Time | best Fitness | total Time |');fprintf('\n')

            hash = zeros(1,32);

            while 1==1   
                [randparam] = randfct(params);
                randparam = condvariablefcn(randparam);
                if xconstraint(randparam,default_parameter)==0
                    continue
                end
    
                hash_ = hash_fct(randparam);
                if sum(hash == hash_,2) == 32
                    b = b+1;
                    continue
                else
                    hash(i,:) = hash_;
                end
                
                tic
                
                [fitness(i), ~, mdl] = fitfun(randparam, default_parameter, self.model_obj, rand_opts);

                self.result(model_num).time(i) = toc;

                if i==1
                    self.result(model_num).total_time(i) = self.result(model_num).time(1);
                else
                    self.result(model_num).total_time(i) = sum(self.result(model_num).total_time(1:i-1));
                end

                fprintf('| %d | %1.4f | %6.1f | %1.4f | %6.1f |',i,fitness(i),self.result(model_num).time(i),min(fitness),self.result(model_num).total_time(i));fprintf('\n');

                self.result(model_num).parameter_all(i,:) = randparam;
                
                self = trace_calculater(self, mdl, rand_opts, parameter, i, model_num);

                if sum(fitness(i)>fitness)==0
                    %self.result(model_num).best_mdl = mdl;
                    self.result(model_num).best_parameter = randparam;
                end

                i = i+1; 

                if self.result(model_num).total_time(end) >= rand_opts.max_time
                    break
                elseif i-1 >= rand_opts.max_obj_eval
                    break
                end
            end
            
            parameter.partition{2}=[];
            parameter.repeats = 1;    
            if isfield(parameter,'validationpatience');parameter.validationpatience=parameter.validationpatience*2;end
            if isfield(parameter,'maxepochs');parameter.maxepochs=parameter.maxepochs*2;end
            parameter.saveNet=1;
            
            parameter.nrepeats = 1; 
            parameter.savemodel=1;
            
            [self.result(model_num).testObj, ~, best_mdl] = fitfun(self.result(model_num).best_parameter, parameter, self.model_obj, rand_opts);
            %self.result(model_num).mdl_result=best_mdl;
            
            cd(self.pth);
            randmdl = self;
            save(self.filename,'randmdl');
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
        if mod(parameter.(name{i}){1},1) == 0  && mod(parameter.(name{i}){2},1) == 0
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

function [objectiv, constraints, model] = fitfun(params, parameter, model, options)
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

if strcmp(parameter.mode, 'classification')
    if ~isempty(model.results(end).validation)
        if strcmp(options.metric, 'kappa') 
            objectiv = 1 - mean([model.results(end).validation.cohenKappa]);
        elseif strcmp(options.metric, 'fp')
            for n=1:size(model.results(end).val_cm,2)
                objectiv(n) = model.results(end).val_cm{1,n}(2,1)/sum(sum(model.results(end).val_cm{1,n}));
            end
            objectiv = mean(objectiv);
        else
            objectiv = 1 - mean([model.results(end).validation.randAccuracy]);
        end
    else
        if strcmp(options.metric, 'kappa') 
            objectiv = 1 - mean([model.results(end).test.cohenKappa]);
        elseif strcmp(options.metric, 'fp')
            for n=1:size(model.results(end).test_cm,2)
                objectiv(n) = model.results(end).test_cm{1,n}(2,1)/sum(sum(model.results(end).test_cm{1,n}));
            end
            objectiv = mean(objectiv);
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
        parameter.xscale = [1/parameter.scale parameter.scale];
        parameter.yscale = [1/parameter.scale parameter.scale];
    elseif ismember('xscale',names) && ismember('yscale',names)
        parameter.xscale = [1/parameter.xscale parameter.xscale];
        parameter.yscale = [1/parameter.yscale parameter.yscale];        
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

function randmdl = trace_calculater(randmdl, mdl, rand_opts, parameter, i, model_num)

if strcmp(parameter.mode, 'classification')
    if ~strcmp(rand_opts.metric, 'kappa') || strcmp(rand_opts.metric, 'acc')
        rand_opts.metric = 'acc';
    end
    if strcmp(rand_opts.metric, 'kappa')
        if isfield(mdl.results(1),'train')
            if i == 1
                randmdl.train_trace(i,model_num) = mean([mdl.results(1).train.cohenKappa]);
            else
                randmdl.train_trace(i,model_num) = max(randmdl.train_trace(i-1,model_num), mean([mdl.results(i).train.cohenKappa]));
            end
        end
        if isfield(mdl.results(1),'validation')
            if i == 1
                randmdl.val_trace(i,model_num) = mean([mdl.results(1).validation.cohenKappa]);
            else
                randmdl.val_trace(i,model_num) = max(randmdl.val_trace(i-1,model_num), mean([mdl.results(i).validation.cohenKappa]));
            end
        end
         if isfield(mdl.results(1),'test')
            if i == 1
                randmdl.test_trace(i,model_num) = mean([mdl.results(1).test.cohenKappa]);
            else
                randmdl.test_trace(i,model_num) = max(randmdl.test_trace(i-1,model_num), mean([mdl.results(i).test.cohenKappa]));
            end
         end
    elseif strcmp(rand_opts.metric, 'acc')
        if isfield(mdl.results(1),'train')
            if i == 1
                randmdl.train_trace(i,model_num) = mean([mdl.results(1).train.cohenKappa]);
            else
                randmdl.train_trace(i,model_num) = max(randmdl.train_trace(i-1,model_num), mean([mdl.results(i).train.randAccuracy]));
            end
        end
        if isfield(mdl.results(1),'validation')
            if i == 1
                randmdl.val_trace(i,model_num) = mean([mdl.results(1).validation.cohenKappa]);
            else
                randmdl.val_trace(i,model_num) = max(randmdl.val_trace(i-1,model_num), mean([mdl.results(i).validation.randAccuracy]));
            end
        end
         if isfield(mdl.results(1),'test')
            if i == 1
                randmdl.test_trace(i,model_num) = mean([mdl.results(1).test.cohenKappa]);
            else
                randmdl.test_trace(i,model_num) = max(randmdl.test_trace(i-1,model_num), mean([mdl.results(i).test.randAccuracy]));
            end
         end
    end
elseif strcmp(parameter.mode, 'regression')
    if ~strcmp(rand_opts.metric, 'rmse') || strcmp(rand_opts.metric, 'r2')
        rand_opts.metric = 'rmse';
    end
    if strcmp(rand_opts.metric, 'rmse')
        if ~isempty(mdl.results(i).RMSE_train)
            if i == 1
                randmdl.train_trace(i,model_num) = mean([mdl.results(1).RMSE_train]);
            else
                randmdl.train_trace(i,model_num) = min(randmdl.train_trace(i-1,model_num), mean([mdl.results(i).RMSE_train]));
            end
        end
        if ~isempty(mdl.results(i).RMSE_val)
            if i == 1
                randmdl.val_trace(i,model_num) = mean([mdl.results(1).RMSE_val]);
            else
                randmdl.val_trace(i,model_num) = min(randmdl.val_trace(i-1,model_num), mean([mdl.results(i).RMSE_val]));
            end
        end
         if ~isempty(mdl.results(i).RMSE_test)
            if i == 1
                randmdl.test_trace(i,model_num) = mean([mdl.results(1).RMSE_test]);
            else
                randmdl.test_trace(i,model_num) = min(randmdl.test_trace(i-1,model_num), mean([mdl.results(i).RMSE_test]));
            end
         end
    elseif strcmp(rand_opts.metric, 'r2')
        if ~isempty(mdl.results(i).r2_train)
            if i == 1
                randmdl.train_trace(i,model_num) = mean([mdl.results(1).r2_train]);
            else
                randmdl.train_trace(i,model_num) = max(randmdl.train_trace(i-1,model_num), mean([mdl.results(i).r2_train]));
            end
        end
        if ~isempty(mdl.results(i).r2_val)
            if i == 1
                randmdl.val_trace(i,model_num) = mean([mdl.results(1).r2_val]);
            else
                randmdl.val_trace(i,model_num) = max(randmdl.val_trace(i-1,model_num), mean([mdl.results(i).r2_val]));
            end
        end
         if ~isempty(mdl.results(i).r2_test)
            if i == 1
                randmdl.test_trace(i,model_num) = mean([mdl.results(1).r2_test]);
            else
                randmdl.test_trace(i,model_num) = max(randmdl.test_trace(i-1,model_num), mean([mdl.results(i).r2_test]));
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

function [param] = randfct(parameter)
param = table();  
names = fieldnames(parameter);
for i=1:length(names)
    param.(names{i}) = eval_optVar(parameter.(names{i}));
    if strcmp(parameter.(names{i}), 'sg_points') == 1
        while mod(param.sg_points,2) == 0 ||  param.sg_points <= param.sg_degree
            param.(names{i}) = eval_optVar(parameter.(names{i}));
        end
    end
end
end


function value = eval_optVar(optVar)
	if strcmp(optVar.Type,'categorical') == 1
        value = categorical(optVar(1).Range);
        idx = randperm(numel(value));
        value = value(idx(1));
    elseif strcmp(optVar.Type,'integer') == 1
        if strcmp(optVar.Transform,'none') == 1
            value = randi([optVar.Range(1) optVar.Range(2)],1,1);
        elseif strcmp(optVar.Transform,'log') == 1
            value = logspace(log10(optVar.Range(1)), log10(optVar.Range(2)),10000);
            idx = randperm(numel(value));
            value = round(value(idx(1)));
        end
	elseif strcmp(optVar.Type,'real') == 1
        if strcmp(optVar.Transform,'none') == 1
            value = optVar.Range(1) + (optVar.Range(2)-optVar.Range(1))*rand(1,1);
        elseif strcmp(optVar.Transform,'log') == 1
            value = logspace(log10(optVar.Range(1)), log10(optVar.Range(2)),10000);
            idx = randperm(numel(value));
            value = value(idx(1));
        end   
	 end
end

function [hash] = hash_fct(parameter)
sha256hasher = System.Security.Cryptography.SHA256Managed;
names = fieldnames(parameter);
hash = [];
for i=1:width(parameter)
    if isa(parameter.(names{i}),'categorical') == 1
        hash = [hash char(parameter.(names{i}))];
    elseif isa(parameter.(names{i}),'double') == 1   
        hash = [hash num2str(round(parameter.(names{i}),3))];
    end
end
hash = uint8(sha256hasher.ComputeHash(uint8(hash)));
end
