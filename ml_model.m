classdef ml_model < hypercube_ensemble
    % spectral-angle-mapper model using hypercubes / hypercube-ensembles
    
    properties
        classes                     % name of the reference classes
        models                      % trained nets
        results                     % net resuts
        model_options
        training_options
        class_id
        class_spectra
        class_scores
        labels
        mean_spek
        mean_labels
        mean_score
        blob_data
        zscore
        pcs
        test_idx
    end
    
    methods
        function mlmdl = ml_model(hc_ensemble, classes, options, exp_name, pth)
            %
%             if exist('hc_ensemble','var') ~= 1
%                 hc_ensemble = mlmdl;
%             end
            mlmdl.hypercube_list = hc_ensemble.hypercube_list;   
            mlmdl.filename = hc_ensemble.filename;              
            mlmdl.pc  = hc_ensemble.pc;                     
            mlmdl.mean_data = hc_ensemble.mean_data;              
            mlmdl.explained = hc_ensemble.explained;                   
            mlmdl.gTruth = hc_ensemble.gTruth;                    
            mlmdl.score_minmax = hc_ensemble.score_minmax; 
            mlmdl.history = hc_ensemble.history;                    
            mlmdl.function_queue = hc_ensemble.function_queue;
            
            mlmdl.model_options{1} = [];
            mlmdl.training_options{1} = [];
            
            if nargin < 5 || isempty(pth); pth = cd; end
            if nargin < 4 || isempty(exp_name)
                time = clock; 
                exp_name = strcat('ML-Model_',num2str(time(1)),'-',num2str(time(2)),...
                    '-',num2str(time(3)),'-',num2str(time(4)),'-',num2str(time(5)));
            end
            addpath(strcat(pth,'\',exp_name));
            mlmdl.filename = exp_name;
            if nargin == 3
                i = size(mlmdl.image_options,1);
                mlmdl.model_options{1} = image_opt(options);
                mlmdl.training_options{1} = augment_opt(options);
            end
            hc = load(mlmdl.hypercube_list{1});
            tmp = fieldnames(hc);
            hc = hc.(tmp{1}); 
            if ~(isempty(hc.label_image))
                classes_ = categories(hc.label_image);
            elseif ~isempty(mlmdl.gTruth)
                classes_ = mlmdl.gTruth;
            else
                classes_ = classes;
            end
            if nargin < 2 || isempty(classes)
                class_id = 1:numel(classes_);
            elseif ~isempty(classes)
                l = 1;
                for i = 1:numel(classes_)
                    for n = 1:numel(classes)
                        if strcmp(classes_{i},classes{n})
                            class_id(l) = n;
                            l = l+1;
                        end
                    end
                end
                classes_ = classes;
            end
            mlmdl.classes = classes_;
            mlmdl.class_id = class_id;
            
            % Erstellen der zusammengesetzten Klassennamen für Klassen mit
            % mehr als einer Klasse
            clear classes_
            for i = 1:length(classes)
                temp = [];
                for l = 1:length(classes{i})
                    temp = strcat(temp,classes{i}(l));
                end
                classes_{i}=temp;
                if iscell(classes_{i})
                    classes_{i} = classes_{i}{1};
                end
            end
            
            for i = 1:numel(hc_ensemble.hypercube_list)
                hc = load(hc_ensemble.hypercube_list{i});
                fprintf('Hypercube %i of %i read \n',i, numel(hc_ensemble.hypercube_list))
                tmp = fieldnames(hc);
                hc = hc.(tmp{1});
                in = 1;
                if ~isempty(hc.label)
                    in = 0;
                    for l = 1:length(classes)
                        idx = strcmp(classes{l},hc.label);
                        for m = 1:length(idx)
                            if idx(m)
                                in = 1;
                                id = l;
                            end
                        end
                    end
                    if in == 0
                        mlmdl.hypercube_list{i} = [];
                    end
                end
                if in
                    mlmdl.mean_spek = vertcat(mlmdl.mean_spek,hc.mean_spek);
                    if ~isempty(hc.label)
                        mlmdl.mean_labels = vertcat(mlmdl.mean_labels, categorical(cellstr(classes_{id})));
                    end
                    if ~(isempty(hc.scores))
                            mlmdl.mean_score = vertcat(mlmdl.mean_score, mean(reshape(hc.scores,hc.samples*hc.lines,size(hc.scores,3))));
                    end
                    if ~isempty(hc.label_image)
                        for n = 1:numel(mlmdl.classes)
                            idx = hc.label_image == categorical(cellstr(mlmdl.classes{n}));
                            speks = reshape(hc.data,hc.samples*hc.lines,hc.bands);
                            if ~(isempty(hc.scores))
                                scores = reshape(hc.scores,hc.samples*hc.lines,size(hc.scores,3));
                            end
                            mlmdl.class_spectra = vertcat(mlmdl.class_spectra,speks(idx(:),:));
                            mlmdl.class_scores = vertcat(mlmdl.class_scores,scores(idx(:),:));
                            mlmdl.labels = vertcat(mlmdl.labels, ones(size(speks(idx(:),:),1),1)*n);
                        end
                    end
                    if ~isempty(hc.blob_data)
                        mlmdl.blob_data = vertcat(mlmdl.blob_data, hc.blob_data);
                    end
                end
            end
            
            for i=1:numel(mlmdl.hypercube_list)
                if isempty(mlmdl.hypercube_list{i})
                    idx(i,1) = true;
                end
            end
            mlmdl.classes = classes_;
            %mlmdl.hypercube_list(idx) = [];
            
            cd(pth)
            mkdir(exp_name)
            cd(exp_name)
            mlmdl.pth = cd();
            save(exp_name,'mlmdl');
            addpath(strcat(pth,'\',exp_name));
        end     
        
        function self = train_model(self, model_num, options)
            if nargin < 3 && isempty(model_num) || nargin < 2
                model_num = size(self.results,2) + 1;
            end
            if nargin == 3 && isempty(model_num)
                i = size(self.results,2);
                self.model_options{i+1,1} = model_opt(options);
                self.training_options{i+1,1} = train_opt_ml(options);
            elseif nargin == 3 && ~(isempty(model_num))
                self.model_options{model_num,1} = model_opt(options);
                self.training_options{model_num,1} = train_opt_ml(options);    
            elseif nargin == 2 && ~isempty(self.model_options(model_num,1)) && ~isempty(self.training_options(model_num,1))
                % nix
            end 
            if isempty(model_num)
                model_num = size(self.results,2) + 1;
            end
            modelopts = self.model_options{model_num,1};
            trainopts = self.training_options{model_num,1};           
            
            ycm = [];
            for i = 1: trainopts.nrepeats
                if strcmp(trainopts.data, 'spectra') || strcmp(trainopts.data, 'data') || strcmp(trainopts.data, 'scores')
                    x = self.class_spectra;
                    y = self.labels;
                elseif strcmp(trainopts.data, 'meanspectra') || strcmp(trainopts.data, 'meanscores')
                    x = self.mean_spek;
                    y = self.mean_labels;
                else
                    error('Wrong data type choosen. Choose "spectra", "scores", "meanscore" or "meanspectra" instead.')
                end
                
                [x, self.pcs{model_num, i}] = datapreprocessor(x, self, trainopts, modelopts);
                
                if modelopts.add_data
                    x = horzcat(x, self.blob_data);
                end
                
                   
                % model training
                tic
                disp(strcat('Modell_',num2str(model_num),'   Versuch: ', num2str(i)))
                
                cd(self.pth);
                mkdir(strcat('Modell_',num2str(model_num)));
                cd(strcat('Modell_',num2str(model_num)));
                mkdir(strcat('Versuch_',num2str(i)));
                cd(strcat('Versuch_',num2str(i)));
                
                % split data
                xtrain = []; xval = []; xtest = [];
                if strcmp(trainopts.validation, 'partition')
                    if length(trainopts.partition{1}) == 1
                        [mlmdl, xtrain, xval, xtest, ytrain, yval, ytest] = split_data(self, x, y, trainopts);
                    else
                        xtest = x(trainopts.partition{1}==1,:);
                        x(trainopts.partition{1}==1,:) = [];
                        ytest = y(trainopts.partition{1}==1,:);
                        y(trainopts.partition{1}==1,:) = [];
                        if ~isempty(trainopts.partition{2})
                            xval = x(trainopts.partition{2}==i,:);
                            xtrain = x(trainopts.partition{2}~=i,:);
                            yval = y(trainopts.partition{2}==i,:);
                            ytrain = y(trainopts.partition{2}~=i,:); 
                        else
                            xtrain = x;
                            ytrain = y; 
                            yval=[];
                            xval=[];
                        end
                    end
                elseif strcmp(trainopts.validation, 'cv')
                    trainopts.partition{2} = 0;
                    [self, xtrain, ~, xtest, ytrain, ~, ytest] = split_data(self, x, y, trainopts);
                    xval = [];
                    yval = [];
                end
                
                %balance data
                %[xtrain, ytrain] = databalancer(xtrain, ytrain);
                
                [self, mdl,cm_train, cm_val, cm_test, ycm_, r2_train, r2_val, r2_test, zs]=trainmdl(self, trainopts, modelopts, xtrain, ytrain, xval, yval, xtest, ytest);         
                ycm = vertcat(ycm, ycm_);
                self.zscore{model_num,i} = zs; 
                
                disp('benötigte Zeit:');
                self.results(model_num).time(i) = toc;
                disp(self.results(model_num).time(i));
                fprintf('\n')
                
                % test net
                if strcmp(modelopts.mode, 'classification')
                    self.results(model_num).train_cm{i} = cm_train;
                    self.results(model_num).train(i) = BM(cm_train);

                    if ~(isempty(cm_val))
                        self.results(model_num).val_cm{i} = cm_val;
                        self.results(model_num).validation(i) = BM(cm_val);
                    end
                    if ~(isempty(cm_test))
                        self.results(model_num).test_cm{i} = cm_test;
                        self.results(model_num).test(i) = BM(cm_test);
                    end

%                     mkdir('Images');
%                     cd('Images');
%                     draw_images(self, self.models{model_num,i}, trainopts)
%                     cd('..');
                elseif strcmp(modelopts.mode, 'regression')
                    self.results(model_num).predError_train{i} = cm_train;
                    self.results(model_num).RMSE_train(i) = sqrt(mean(cm_train.^2));
                    self.results(model_num).r2_train(i) = mean(r2_train);
                    if ~(isempty(cm_val))
                        self.results(model_num).predError_val{i} = cm_val;
                        self.results(model_num).RMSE_val(i) = sqrt(mean(cm_val.^2));
                        self.results(model_num).r2_val(i) = mean(r2_val);
                    end
                    if ~(isempty(cm_test))
                        self.results(model_num).predError_test{i} = cm_test;
                        self.results(model_num).RMSE_test(i) = sqrt(mean(cm_test.^2));
                        self.results(model_num).r2_test(i) = mean(r2_test);
                    end

                end
                
                if modelopts.save_model
                    %self(model_num).models{i} = mdl;
                    save('Model','mdl');
                end
            end
            
            if strcmp(modelopts.mode, 'classification')
                cd('..')
                draw_confusionmat(ycm(:,1),ycm(:,2), self.classes, 'mean');
            elseif strcmp(modelopts.mode, 'regression')
                cd('..')
                %ycm = round(mean(ycm,3));
                draw_regression(ycm(:,1),ycm(:,2), 'mean', 1) 
            end
            
            % save seg model
            cd(self.pth);
            mlmdl = self;
            save(self.filename,'mlmdl');
        end
        
        function export_classimage(self, model, data, classes, dpi, pth)
            % exports images (hc.(data)(range) with the Classifikation-Overlay for each
            % class. 
            
            if nargin < 4 || isempty(dpi); dpi = 100; end
            if nargin < 3 || isempty(classes)
                for i = 1: length(self.classes)
                    class{i} = self.classes{i};
                end  
            else
                for i = 1: length(classes)
                    for n = 1: size(self.classes,2)
                        if strcmp(classes{i}, self.classes{1,n})
                            class{1,i} = self.classes{1,n};
                        end
                    end
                end
            end         
            if nargin < 3; error('Not enough input arguments'); end
            
            if length(model) == 2
                if ~isempty(self(model(1)).models{model(2)})
                    mdl = self(model(1)).models{model(2)};
                    train_opts = self.training_options{model(1)};
                else
                    error('Choose a valid model!');
                end
            elseif length(model) == 1
                if isfield(self.results(model(1)),'test')
                    acc = self.results(model(1)).test.randAccuracy;
                    [~, model(2)] = max(acc);
                elseif ~isempty(self.results(model(1)).validation)
                    acc = self.results(model(1)).validation.randAccuracy;
                    [~, model(2)] = max(acc);
                elseif ~isempty(self.results(model(1)).train)
                    acc = self.results(model(1)).train.randAccuracy;
                    [~, model(2)] = max(acc);
                end
                if ~isempty(self(model(1)).models{model(2)})
                    mdl = self(model(1)).models{model(2)};
                    train_opts = self.training_options{model(1)};
                else
                    error('Choose a valid model!');
                end
            elseif isempty(model)
                error('Choose a valid model!');
            end
            
            if nargin < 5 || isempty(pth); pth = strcat(self.pth,'\Modell_', num2str(model(1)), '\Versuch_', num2str(model(2))); end
            
            for i = 1:numel(self.hypercube_list)
                clear overlay
                hc = load(self.hypercube_list{i});
                                
                fprintf('\nHypercube %i of %i read \n',i, numel(self.hypercube_list))
                
                tmp = fieldnames(hc);
                hc = hc.(tmp{1});
                clear y; 
                x = reshape(hc.data,hc.samples*hc.lines,size(hc.data,3));
                if strcmp(train_opts.data, 'scores') || strcmp(train_opts.data, 'meanscores')
                    t = (self.pcs{model});
                	x=x * t;
                end
                if train_opts.zscore
                    x = (x-repmat(self.zscore{model(1),model(2)}(1,:),[size(x,1),1]))./repmat(self.zscore{model(1),model(2)}(2,:),[size(x,1),1]);
                end
                tic
                if strcmp(train_opts.validation, 'cv')
                    for n = 1:length(mdl.Trained)
                        y(:,i) = predict(mdl.Trained{i},x);
                    end
                y = mode(y,2);
                else
                    y = predict(mdl, x);
                end
                toc
                y = reshape(y, hc.samples, hc.lines, 1);
                y = categorical(y,self.class_id,self.classes); 
                if strcmp(data, 'scores')
                    data_ = hc.scores(:,:,1);
                elseif strcmp(data, 'data')
                    data_ = hc.sb;
                end
                
                class_names = '';
                for n = 1: length(class)
                    class_names = strcat(class_names,'_',class{n});
                end
                
                name = strcat('_',data, '_Modell-', strrep(num2str(model),'  ','-'),'_ClassImage',class_names);
                hc.export_image(data_,[], 1, y, dpi, pth, name);
            end   
        end  
    end
end

%--------------------------------------------------------------------------
% supporting functions
function [mlmdl, mdl, cm_train, cm_val, cm_test, ycm, r2_train, r2_val, r2_test, zs] = trainmdl(mlmdl, trainopts, modelopts, xtrain, ytrain, xval, yval, xtest, ytest)
ycm = [];

if trainopts.zscore
    [xtrain,mu,sigma] = zscore(xtrain);
    if ~isempty(xval)
        xval = (xval-repmat(mu,[size(xval,1),1]))./repmat(sigma,[size(xval,1),1]);
    end
    if ~isempty(xtest)
        xtest = (xtest-repmat(mu,[size(xtest,1),1]))./repmat(sigma,[size(xtest,1),1]);
    end
    zs = vertcat(mu, sigma);
end


if strcmp(modelopts.mode, 'classification')
    if strcmp(modelopts.model, 'da')
        if ~strcmp(modelopts.da_discrimtyp, 'linear')
            modelopts.da_delta = 0;
            modelopts.da_gamma = 1;
        end
        if strcmp(trainopts.validation, 'cv')
            mdl = fitcdiscr(xtrain,ytrain,'KFold',trainopts.nkfold,'DiscrimType',modelopts.da_discrimtyp,'Delta',modelopts.da_delta,'Gamma',modelopts.da_gamma);
        elseif strcmp(trainopts.validation, 'partition')
            mdl = fitcdiscr(xtrain,ytrain,'DiscrimType',modelopts.da_discrimtyp,'Delta',modelopts.da_delta,'Gamma',modelopts.da_gamma);
        end
    end

    if strcmp(modelopts.model, 'knn')
        if strcmp(modelopts.knn_distance,'Minkowski')
            if strcmp(trainopts.validation, 'cv')
                mdl = fitcknn(xtrain,ytrain,'KFold',trainopts.nkfold,'NumNeighbors',modelopts.knn_numneigh,'Standardize',modelopts.knn_standardize,'Distance',modelopts.knn_distance,'DistanceWeight',modelopts.knn_distanceweight,'Exponent',modelopts.knn_exponent);
            elseif strcmp(trainopts.validation, 'partition')
                mdl = fitcknn(xtrain,ytrain,'NumNeighbors',modelopts.knn_numneigh,'Standardize',modelopts.knn_standardize,'Distance',modelopts.knn_distance,'DistanceWeight',modelopts.knn_distanceweight,'Exponent',modelopts.knn_exponent);
            end
        else
            if strcmp(trainopts.validation, 'cv')
                mdl = fitcknn(xtrain,ytrain,'KFold',trainopts.nkfold,'NumNeighbors',modelopts.knn_numneigh,'Standardize',(modelopts.knn_standardize),'Distance',char(modelopts.knn_distance),'DistanceWeight',char(modelopts.knn_distanceweight));      
            elseif strcmp(trainopts.validation, 'partition')
                mdl = fitcknn(xtrain,ytrain,'NumNeighbors',modelopts.knn_numneigh,'Standardize',(modelopts.knn_standardize),'Distance',char(modelopts.knn_distance),'DistanceWeight',char(modelopts.knn_distanceweight));      
            end
        end
    end

    if strcmp(modelopts.model, 'svm')
        if strcmp(modelopts.svm_kernel, 'polynomial')
            learner =  templateSVM('PolynomialOrder',modelopts.svm_polynom,'BoxConstraint',modelopts.svm_box,'KernelScale',modelopts.svm_kernelsc,'Standardize',(modelopts.svm_standardize),'KernelFunction',char(modelopts.svm_kernel));
            if strcmp(trainopts.validation, 'cv')
                mdl = fitcecoc(xtrain,ytrain,'KFold',trainopts.nkfold,'Coding','onevsall','Learners',learner);            
            elseif strcmp(trainopts.validation, 'partition')
                mdl = fitcecoc(xtrain,ytrain,'Coding','onevsall','Learners',learner);                
            end            
        else
            learner =  templateSVM('BoxConstraint',modelopts.svm_box,'KernelScale',modelopts.svm_kernelsc,'Standardize',(modelopts.svm_standardize),'KernelFunction',char(modelopts.svm_kernel));
            if strcmp(trainopts.validation, 'cv')
                mdl = fitcecoc(xtrain,ytrain,'KFold',trainopts.nkfold,'Coding','onevsall','Learners',learner);
            elseif strcmp(trainopts.validation, 'partition')
                mdl = fitcecoc(xtrain,ytrain,'Coding','onevsall','Learners',learner);               
            end  
        end
        clear learner
    end

    if strcmp(modelopts.model, 'ensemble')
        if length(unique(ytrain)) == 2 && strcmp(modelopts.ensemble_methode, 'AdaBoost')
            ensemble_methode=strcat(char(modelopts.ensemble_methode),'M1');
            learner = templateTree('MaxNumSplits',modelopts.ensemble_MaxNumSplits,'MinLeafSize',modelopts.ensemble_MinLeafSize,'NumVariablesToSample',modelopts.ensemble_NumVariablesToSample);
            if strcmp(trainopts.validation, 'cv')
                mdl = fitcensemble(xtrain,ytrain,'KFold',trainopts.nkfold,'Learners',learner,'Method',ensemble_methode,'NumLearningCycles',modelopts.ensemble_NumLearningCycles,'LearnRate',modelopts.ensemble_LearnRate);
            elseif strcmp(trainopts.validation, 'partition')
                mdl = fitcensemble(xtrain,ytrain,'Learners',learner,'Method',ensemble_methode,'NumLearningCycles',modelopts.ensemble_NumLearningCycles,'LearnRate',modelopts.ensemble_LearnRate);
            end
        elseif length(unique(ytrain)) >= 3 && strcmp(modelopts.ensemble_methode, 'AdaBoost')
            ensemble_methode=strcat(char(modelopts.ensemble_methode),'M2');
            learner = templateTree('MaxNumSplits',modelopts.ensemble_MaxNumSplits,'MinLeafSize',modelopts.ensemble_MinLeafSize,'NumVariablesToSample',modelopts.ensemble_NumVariablesToSample);
            if strcmp(trainopts.validation, 'cv')
                mdl = fitcensemble(xtrain,ytrain,'KFold',trainopts.nkfold,'Learners',learner,'Method',ensemble_methode,'NumLearningCycles',modelopts.ensemble_NumLearningCycles,'LearnRate',modelopts.ensemble_LearnRate);
            elseif strcmp(trainopts.validation, 'partition')
                mdl = fitcensemble(xtrain,ytrain,'Learners',learner,'Method',ensemble_methode,'NumLearningCycles',modelopts.ensemble_NumLearningCycles,'LearnRate',modelopts.ensemble_LearnRate);
            end
        else
            if strcmp(modelopts.ensemble_methode, 'Bag')
                learner = templateTree('MaxNumSplits',modelopts.ensemble_MaxNumSplits,'MinLeafSize',modelopts.ensemble_MinLeafSize,'NumVariablesToSample',modelopts.ensemble_NumVariablesToSample);
                if strcmp(trainopts.validation, 'cv')
                    mdl = fitcensemble(xtrain,ytrain,'KFold',trainopts.nkfold,'Learners',learner,'Method',char(modelopts.ensemble_methode),'NumLearningCycles',modelopts.ensemble_NumLearningCycles);
                elseif strcmp(trainopts.validation, 'partition')
                    mdl = fitcensemble(xtrain,ytrain,'Learners',learner,'Method',char(modelopts.ensemble_methode),'NumLearningCycles',modelopts.ensemble_NumLearningCycles);
                end
            else
                learner = templateTree('MaxNumSplits',modelopts.ensemble_MaxNumSplits,'MinLeafSize',modelopts.ensemble_MinLeafSize,'NumVariablesToSample',modelopts.ensemble_NumVariablesToSample);
                if strcmp(trainopts.validation, 'cv')
                    mdl = fitcensemble(xtrain,ytrain,'KFold',trainopts.nkfold,'Learners',learner,'Method',char(modelopts.ensemble_methode),'NumLearningCycles',modelopts.ensemble_NumLearningCycles,'LearnRate',modelopts.ensemble_LearnRate);
                elseif strcmp(trainopts.validation, 'partition')
                    mdl = fitcensemble(xtrain,ytrain,'Learners',learner,'Method',char(modelopts.ensemble_methode),'NumLearningCycles',modelopts.ensemble_NumLearningCycles,'LearnRate',modelopts.ensemble_LearnRate);
                end
            end
         end
        clear learner
    end
    
    if strcmp(modelopts.model, 'nn')
        
        if (isa(ytrain,'categorical'))
            ytrain_ = double(ytrain);
            %yval = double(yval);
            %ytest = double(ytest);
        end
        
        ytrain_ = (full(ind2vec(ytrain_')));
        
        layer = [modelopts.nn_hiddensize1 modelopts.nn_hiddensize2 modelopts.nn_hiddensize3];
        layer(layer==0) = [];
        if strcmp(trainopts.validation, 'cv')
            idx = randperm(size(xtrain,1));
            nums = floor(length(idx)/trainopts.nkfold);
            for i = 1:trainopts.nkfold
                mdl{i} = patternnet(layer, modelopts.nn_fitfun);
                idx_ = idx; idx_(1+(i-1)*nums:i*nums) = []; idx_(end-(length(idx)-nums*trainopts.nkfold)+1:end)=[];
                x_ = xtrain(idx_,:)';
                y_ = ytrain_(:,idx_);
                mdl{i} = train(mdl{i},x_,y_,'useParallel','yes');
                nntraintool close
            end
        elseif strcmp(trainopts.validation, 'partition')
            mdl = patternnet(layer, modelopts.nn_fitfun);
            mdl = train(mdl,xtrain',ytrain_,'useParallel','yes');
            nntraintool close
        end  
    end
        
elseif strcmp(modelopts.mode, 'regression')
    if ~strcmp(modelopts.model, 'ensemble')&&~strcmp(modelopts.model, 'grnn')&&~strcmp(modelopts.model, 'nn')&&~strcmp(modelopts.model, 'svm')
        modelopts.model = 'ensemble';
        fprintf('\n Model changed to "ensemble".\n')
    end
    
    if ~(isa(ytrain,'double'))
        ytrain = str2num(ytrain);
        yval = str2num(yval);
        ytest = str2num(ytest);
    end
    
    if strcmp(modelopts.model, 'ensemble')
        if ~strcmp(modelopts.ensemble_methode, 'Bag') && ~strcmp(modelopts.ensemble_methode, 'LSBoost')
            modelopts.ensemble_methode = 'Bag';
        end
            if strcmp(modelopts.ensemble_methode, 'Bag')
                learner = templateTree('MaxNumSplits',modelopts.ensemble_MaxNumSplits,'MinLeafSize',modelopts.ensemble_MinLeafSize,'NumVariablesToSample',modelopts.ensemble_NumVariablesToSample);
                if strcmp(trainopts.validation, 'cv')
                    mdl = fitrensemble(xtrain,ytrain,'KFold',trainopts.nkfold,'Learners',learner,'Method',char(modelopts.ensemble_methode),'NumLearningCycles',modelopts.ensemble_NumLearningCycles);
                elseif strcmp(trainopts.validation, 'partition')
                    mdl = fitrensemble(xtrain,ytrain,'Learners',learner,'Method',char(modelopts.ensemble_methode),'NumLearningCycles',modelopts.ensemble_NumLearningCycles);
                end
            elseif strcmp(modelopts.ensemble_methode, 'LSBoost')
                learner = templateTree('MaxNumSplits',modelopts.ensemble_MaxNumSplits,'MinLeafSize',modelopts.ensemble_MinLeafSize,'NumVariablesToSample',modelopts.ensemble_NumVariablesToSample);
                if strcmp(trainopts.validation, 'cv')
                    mdl = fitrensemble(xtrain,ytrain,'KFold',trainopts.nkfold,'Learners',learner,'Method',char(modelopts.ensemble_methode),'NumLearningCycles',modelopts.ensemble_NumLearningCycles,'LearnRate',modelopts.ensemble_LearnRate);
                elseif strcmp(trainopts.validation, 'partition')
                    mdl = fitrensemble(xtrain,ytrain,'Learners',learner,'Method',char(modelopts.ensemble_methode),'NumLearningCycles',modelopts.ensemble_NumLearningCycles,'LearnRate',modelopts.ensemble_LearnRate);
                end
            end
        clear learner
    elseif strcmp(modelopts.model, 'svm')
        if strcmp(modelopts.svm_kernel, 'polynomial')
            if strcmp(trainopts.validation, 'cv')
                mdl = fitrsvm(xtrain,ytrain,'KFold',trainopts.nkfold,'PolynomialOrder',modelopts.svm_polynom,'BoxConstraint',modelopts.svm_box,'KernelScale',modelopts.svm_kernelsc,'Standardize',(modelopts.svm_standardize),'KernelFunction',char(modelopts.svm_kernel));            
            elseif strcmp(trainopts.validation, 'partition')
                mdl = fitrsvm(xtrain,ytrain,'PolynomialOrder',modelopts.svm_polynom,'BoxConstraint',modelopts.svm_box,'KernelScale',modelopts.svm_kernelsc,'Standardize',(modelopts.svm_standardize),'KernelFunction',char(modelopts.svm_kernel));                
            end            
        else
            if strcmp(trainopts.validation, 'cv')
                mdl = fitrsvm(xtrain,ytrain,'KFold',trainopts.nkfold,'BoxConstraint',modelopts.svm_box,'KernelScale',modelopts.svm_kernelsc,'Standardize',(modelopts.svm_standardize),'KernelFunction',char(modelopts.svm_kernel));
            elseif strcmp(trainopts.validation, 'partition')
                mdl = fitrsvm(xtrain,ytrain,'BoxConstraint',modelopts.svm_box,'KernelScale',modelopts.svm_kernelsc,'Standardize',(modelopts.svm_standardize),'KernelFunction',char(modelopts.svm_kernel));               
            end  
        end
        clear learner
    elseif strcmp(modelopts.model, 'grnn')
        if strcmp(trainopts.validation, 'cv')
            idx = randperm(size(xtrain,1));
            nums = floor(length(idx)/trainopts.nkfold);
            for i = 1:trainopts.nkfold
                idx_ = idx; idx_(1+(i-1)*nums:i*nums) = []; idx_(end-(length(idx)-nums*trainopts.nkfold)+1:end)=[];
                x_ = double(xtrain(idx_,:)');
                y_ = double(ytrain(idx_)');
                mdl{i} = newgrnn(x_,y_,modelopts.grnn_spread);
            end
        elseif strcmp(trainopts.validation, 'partition')
            mdl = newgrnn(double(xtrain'),double(ytrain'),modelopts.grnn_spread);
        end
    elseif strcmp(modelopts.model, 'nn')
        layer = [modelopts.nn_hiddensize1 modelopts.nn_hiddensize2 modelopts.nn_hiddensize3];
        layer(layer==0) = [];
        if strcmp(trainopts.validation, 'cv')
            idx = randperm(size(xtrain,1));
            nums = floor(length(idx)/trainopts.nkfold);
            for i = 1:trainopts.nkfold
                idx_ = idx; idx_(1+(i-1)*nums:i*nums) = []; idx_(end-(length(idx)-nums*trainopts.nkfold)+1:end)=[];
                x_ = xtrain(idx_,:)';
                y_ = ytrain(idx_)';
                mdl{i} = feedforwardnet(layer, modelopts.nn_fitfun);
                mdl{i} = train(mdl{i},x_,y_,'useParallel','yes');
            end
        elseif strcmp(trainopts.validation, 'partition')
            mdl = feedforwardnet(layer, modelopts.nn_fitfun);
            mdl = train(mdl,xtrain',ytrain','useParallel','yes');
        end        
    end
end

cm_train = []; cm_val = []; cm_test = [];
r2_train = []; r2_val = []; r2_test = [];

if strcmp(modelopts.mode, 'classification')
    if strcmp(trainopts.validation, 'partition')
        if ~strcmp(modelopts.model, 'nn')
            Y = resubPredict(mdl);
        else
            Y = categorical(vec2ind(round(mdl(xtrain')))');
        end
        %draw_confusionmat(ytrain, Y, mlmdl.classes, 'training');
        cm_train = confusionmat(ytrain, Y);
        if ~(isempty(xval))
            if ~strcmp(modelopts.model, 'nn')
                Y = predict(mdl, xval);
            else
                Y = categorical(vec2ind(round(mdl(xval')))');
            end
            
            %draw_confusionmat(yval, Y, mlmdl.classes, 'validation');
            cm_val = confusionmat(yval, Y);
            ycm = [yval, Y];
        end
        if ~(isempty(xtest))
            if ~strcmp(modelopts.model, 'nn')
                Y = predict(mdl, xtest);
            else
                Y = categorical(vec2ind(round(mdl(xval')))');
            end            
            %draw_confusionmat(ytest, Y, mlmdl.classes, 'test');
            cm_test = confusionmat(ytest, Y);
            ycm = [ytest, Y];
        end
    elseif strcmp(trainopts.validation, 'cv')
        if ~strcmp(modelopts.model, 'nn')
            Y = kfoldPredict(mdl);
        else
            Y=[];
            for i = 1:length(mdl)
                idx_ = idx(1+(i-1)*nums:i*nums);
                x_ = xtrain(idx_,:)';
                Y = vertcat(Y,categorical(vec2ind(round(mdl{i}(x_)))'));
            end
            ytrain = ytrain(1:length(Y));
        end
        draw_confusionmat(ytrain, Y, mlmdl.classes, 'cross-validation');
        cm_val = confusionmat(ytrain, Y);
        ycm = [ytrain, Y];
        if ~(isempty(xtest))
            clear Y
            if ~strcmp(modelopts.model, 'nn')
                for i = 1:length(mdl.Trained)
                    Y(:,i) = predict(mdl.Trained{i}, xtest);
                end
            else
                Y=[];
                for i = 1:length(mdl)
                    Y(:,i) = categorical(vec2ind(round(mdl{i}(xtest')))');
                end
            end
            Y = categorical(mode(Y,2));
            draw_confusionmat(ytest, Y, mlmdl.classes, 'test');
            cm_test = confusionmat(ytest, Y);
            ycm = [ytest, Y];
        end
    end
elseif strcmp(modelopts.mode, 'regression')
    if strcmp(trainopts.validation, 'partition')
        if ~strcmp(modelopts.model, 'nn') && ~strcmp(modelopts.model, 'grnn')
            Y = resubPredict(mdl);
        else
            Y = mdl(xtrain')';
        end
        r2_train = draw_regression(ytrain, Y, 'training', 0);
        cm_train = Y-ytrain;
        if ~(isempty(xval))
            if ~strcmp(modelopts.model, 'nn') && ~strcmp(modelopts.model, 'grnn')
                Y = predict(mdl, xval);
            else
                Y = mdl(xval')';
            end
            r2_val = draw_regression(yval, Y, 'validation', 0);
            cm_val = Y-yval;
            ycm = [yval, Y];
        end
        if ~(isempty(xtest))
            if ~strcmp(modelopts.model, 'nn') && ~strcmp(modelopts.model, 'grnn')
                Y = predict(mdl, xtest);
            else
                Y = mdl(xtest')';
            end
            r2_test = draw_regression(ytest, Y, 'test', 0);
            cm_test = Y-ytest;
            ycm = [ytest, Y];
        end
    elseif strcmp(trainopts.validation, 'cv')
        if ~strcmp(modelopts.model, 'nn') && ~strcmp(modelopts.model, 'grnn')
            Y = kfoldPredict(mdl);
        else
            Y=[];
            for i = 1:length(mdl)
                idx_ = idx(1+(i-1)*nums:i*nums);
                x_ = xtrain(idx_,:)';
                Y = vertcat(Y,(mdl{i}(x_))');
            end
            ytrain=ytrain(1:length(Y));
        end
        r2_val = draw_regression(ytrain, Y, 'cross-validation', 1);
        cm_val = ytrain-Y;
        ycm = [ytrain, Y];
        if ~(isempty(xtest))
            clear Y
            if ~strcmp(modelopts.model, 'nn') && ~strcmp(modelopts.model, 'grnn')
                %nummdl = length(mdl.Trained);
                for i = 1:length(mdl.Trained)
                    Y(:,i) = predict(mdl.Trained{i}, xtest);
                end
            else
                Y=[];
                %nummdl = length(mdl);
                for i = 1:length(mdl)
                    Y(:,i) = mdl{i}(xtest');
                end
            end
            Y = mean(Y,2);
            %Y = Y(:);
            %ytest = repmat(ytest,[nummdl 1]);
            r2_test = draw_regression(ytest, Y, 'test', 1);
            cm_test = Y-ytest;
            ycm = [ytest, Y];
        end
    end
end

end

function [mlmdl, xtrain, xval, xtest, ytrain, yval, ytest] = split_data(mlmdl, x, y, options)

idxtest = [];
xtest = [];
ytest = [];

if ~isempty(mlmdl.test_idx)
    options.partition{1} = options.partition{1}/(1-options.partition{3});
    options.partition{2} = options.partition{2}/(1-options.partition{3});
    options.partition{3} = 0;
    idxtest = mlmdl.test_idx;
    xtest = x(idxtest,:);
    ytest = y(idxtest,:);
    y(idxtest) = [];
    x(idxtest,:) = [];
end

numfiles = length(y);
shuffledidx = randperm(numfiles);

% training
n = floor(options.partition{1}*numfiles * options.data_partition);
idxtrain = shuffledidx(1:n);

%validation
m = floor(options.partition{2}*numfiles * options.data_partition);
idxval = shuffledidx(n+1:n+m);

%test
if length(options.partition) == 3
    l = floor(options.partition{3}*numfiles * options.data_partition);
else
    l=0;
end
if isempty(l)
    l=0;
end
if isempty(idxtest) && l ~= 0
    idxtest = shuffledidx(n+m+1:n+m+l);
end

% image datastores
xtrain = x(idxtrain,:);
ytrain = y(idxtrain,:);

if ~(isempty(idxval))
    xval = x(idxval,:);
    yval = y(idxval,:);
else
    xval = [];
    yval = [];
end
if ~(isempty(idxtest)) && isempty(mlmdl.test_idx)
    xtest = x(idxtest,:);
    ytest = y(idxtest,:);
end

if isempty(mlmdl.test_idx)
    mlmdl.test_idx = idxtest;
end

end

function [x, y] = databalancer(x, y)

%make training dataset balanced
unique_labels = unique(y);
max_labels = 0; maxnum_labels = 0;
for i = 1:length(unique_labels)
    idx_labels{i} = find(y == unique_labels(i));
    if length(idx_labels{i}) > maxnum_labels
        maxnum_labels = length(idx_labels{i});
    end
end

xnew = x;
ynew = y;
for i = 1:length(unique_labels)
    idx = [];
    l = length(idx_labels{i});
    n = 1;
    while l < maxnum_labels
        idx(n) = randsample(idx_labels{i},1);
        l = l + 1;
        n = n + 1;
    end
    if ~isempty(idx)
        xnew = vertcat(xnew, x(idx,:));
        ynew = vertcat(ynew, y(idx,:));
    end
end
x=xnew;
y=ynew;

end

function [xtrain, pc] = datapreprocessor(xtrain, mlmdl, trainopts, modelopts)
    
if strcmp(modelopts.noise, 'savitzky-golay')
    xtrain = sgolay_fun(xtrain, modelopts.sg_degree, modelopts.sg_points, modelopts.sg_deriv);
end
if strcmp(modelopts.scaling, 'none')
elseif strcmp(modelopts.scaling, 'SNV')
    xtrain = snv_fun(xtrain);
elseif strcmp(modelopts.scaling, 'minmax')
    xtrain = minmax_fun(xtrain);
elseif strcmp(modelopts.scaling, 'inf')
    xtrain = norm_fun(xtrain, inf);
else    
    if isnumeric(modelopts.scaling)
        xtrain = norm_fun(xtrain, (modelopts.scaling));
    else
        modelopts.scaling=str2num(modelopts.scaling);
        xtrain = norm_fun(xtrain, (modelopts.scaling));
    end
end

if strcmp(trainopts.data, 'scores') || strcmp(trainopts.data, 'meanscores')
    %[pc, xtrain, ~] = pca(xtrain,'Centered', 'off');
    [pc, xtrain, ~] = pca(xtrain);
    xtrain = xtrain(:,trainopts.range);
    pc = pc(:, trainopts.range);
else
    pc = [];
end

% if modelopts.add_data == 1
%     xtrain = horzcat(xtrain, mlmdl.blob_data);
% end

end

function draw_confusionmat(ytrue, ypred, classes, name)

ytrue = grp2idx(ytrue);
ypred = grp2idx(ypred);
numclass = length(classes);
targets = zeros(numclass, length(ytrue));
outputs = zeros(numclass, length(ytrue));
targetidx = sub2ind(size(targets), ytrue', 1: length(ytrue));
outputidx = sub2ind(size(outputs), ypred', 1: length(ytrue));
targets(targetidx) = 1;
outputs(outputidx) = 1;
plotconfusion(targets, outputs);
h = gca;

h.XTickLabel = classes;
h.XTickLabel{numclass+1} = '';
h.YTickLabel = classes;
h.YTickLabel{numclass+1} = '';
h.YTickLabelRotation = 90;
set(gcf,'Position',get(0,'Screensize'));
name = strcat('Confusionmatrix_', name);
%export_fig(gcf,name,'-painters','-png',150);
savefig(strcat(name,'.fig'));
close all
end

function draw_images(mlmdl, mdl, options)
for i = 1:numel(mlmdl.hypercube_list)
    hc = load(mlmdl.hypercube_list{i});
    fprintf('Hypercube %i of %i read \n',i, numel(mlmdl.hypercube_list))
    tmp = fieldnames(hc);
    hc = hc.(tmp{1});
    x = hc.(options.data)(:,:,options.range);
    [s1, s2, s3] = size(x);
    x = reshape(x, s1*s2, s3);
    if strcmp(options.validation, 'cv')
        clear classimg
        for n = 1: length(mdl.Trained)
            im_ = predict(mdl.Trained{n}, x);
            classimg(:,:,n) = reshape(im_,s1, s2, 1);
        end
        classimg = mode(classimg, 3);
    elseif strcmp(options.validation, 'partition')
        classimg = predict(mdl,x);
        classimg = reshape(classimg, s1, s2, 1);
    end
    im = mat2gray(mean(hc.data(:,:,round(0.25*length(hc.wl)): round(0.75*length(hc.wl))),3));
    [~,fn,~]=fileparts(hc.filename);
    a=labeloverlay(im,classimg,'Transparency',0.7);
    imwrite(a,strcat(fn,'_class','.png'));
end
end

function [r2] = draw_regression(ytrue, ypred, name, draw)
mse = mean((ytrue - ypred).^2);
rmse = sqrt(mse);

% R^2
X = [ones(length(ytrue),1) ytrue];
b = X\ypred;
ycalc = X*b;
r2 = 1 - sum((ytrue - ycalc).^2)/sum((ytrue - mean(ytrue)).^2);

if draw
    figure1 = figure('Color',[1 1 1]);
    axes1 = axes('Parent',figure1,...
        'Position',[0.13 0.11 0.403854166666667 0.815]);
    hold(axes1,'on');
    plot1 = plot(ytrue, ytrue,'-k','LineWidth',1);

    plot2 = plot(ytrue, ycalc, '-r','LineWidth',1);
    set(plot1,'DisplayName','reference','Color',[0 0 0]);
    set(plot2,'DisplayName','fitted','Color',[1 0 0]);
    scatter(ytrue, ypred, 25, 'blue', 'filled','DisplayName','data')
    hold(axes1,'on');
    ylabel('predicted','FontWeight','bold');
    xlabel('measured','FontWeight','bold');
    box(axes1,'on');
    set(axes1,'FontSize',14,'FontWeight','bold');
    legend1 = legend(axes1,'show');
    set(legend1,...
        'Position',[0.147066920975016 0.8098389002327 0.0623080790249838 0.099260525518988]);
    legend boxoff
    annotation(figure1,'textbox',...
        [0.141104166666666 0.653029371617516 0.0632812513535222 0.150739479932904],...
        'String',{'MSE =',num2str(mse),'RMSE =',num2str(rmse),'R^2 =',num2str(r2)},...
        'LineStyle','none',...
        'FontWeight','bold',...
        'FontSize',14,'FitBoxToText','off');
    set(gcf,'Position',get(0,'Screensize'));

    name = strcat('RegressionPlot_', name);
    export_fig(gcf,name,'-painters','-png',150);
    savefig(strcat(name,'.fig'));
    close all
end
end

function [hc]=sgolay_fun(hc,polynom,points,deriv)

if ndims(hc)==2
    [s1,s2]=size(hc);s=s1;
    hc_=zeros(s,s2);
elseif ndims(hc)==3
    [s1,s2,s3]=size(hc);
    hc=reshape(hc,s1*s2,s3);
    s=s1*s2;
    hc_=zeros(s,s3);
else
end
   
    [b,g]=sgolay(polynom,points);
    halfwin=((points+1)/2)-1;
    
    if deriv==0
            for n=(points+1)/2:length(hc(1,:))-(points+1)/2
                hc_(:,n)=dot(repmat(g(:,1)',s,1),hc(:,n-halfwin:n+halfwin),2);
            end
    elseif deriv==1
            for n=(points+1)/2:length(hc(1,:))-(points+1)/2
                hc_(:,n)=dot(repmat(g(:,2)',s,1),hc(:,n-halfwin:n+halfwin),2);
            end
    elseif deriv==2
            for n=(points+1)/2:length(hc(1,:))-(points+1)/2
                hc_(:,n)=dot(repmat(g(:,3)',s,1),hc(:,n-halfwin:n+halfwin),2);
            end
    end

hc_=hc_(:,ceil(points/2):end-ceil(points/2));
    
if exist('s3')==1
    hc=reshape(hc_,s1,s2,s3-points);
else
    hc=hc_;
end
end

function [hc]=norm_fun(hc, pnorm)
if ndims(hc)==2
    [~,s2]=size(hc);
    if isinf(pnorm) ~= 1
        norms=repmat((sum(abs(hc).^pnorm,2).^(1/pnorm)),[1,s2]);
    else
        norms = repmat(max(abs(hc),[],2),[1, s2]);
    end
    hc((sum(isnan(hc),2)) ~= 0,:) = 0;
    hc=hc./norms;
elseif ndims(hc)==3
    [s1,s2,s3]=size(hc);
    hc=reshape(hc,s1*s2,s3);
    if isinf(pnorm) ~= 1
        norms=repmat((sum(abs(hc).^pnorm,2).^(1/pnorm)),[1,s3]);
    else
        norms = repmat(max(abs(hc),[],2),[1, s3]);
    end
    hc=hc./norms;
    hc((sum(isnan(hc),2)) ~= 0,:) = 0;
    hc=reshape(hc,s1,s2,s3);
else
end
end

function [hc]=snv_fun(hc)
if ndims(hc)==2
    [~,s2]=size(hc);
    mean_hc=mean(hc,2);
    std_hc=std(hc,[],2);
    hc=(hc-repmat(mean_hc,[1,s2]))./repmat(std_hc,[1,s2]);
elseif ndims(hc)==3
    [s1,s2,s3]=size(hc);
    hc=reshape(hc,s1*s2,s3);
    mean_hc=mean(hc,2);
    std_hc=std(hc,[],2);
    hc=(hc-repmat(mean_hc,[1,s3]))./repmat(std_hc,[1,s3]);
    hc=reshape(hc,s1,s2,s3);
end
end

function [hc] = minmax_fun(hc)
    if ndims(hc)==2
        [~,s2]=size(hc);
        hc=(hc-repmat(min(hc,[],2),[1,s2]))./repmat(max(hc,[],2)-min(hc,[],2),[1,s2]);
    elseif ndims(hc)==3
        [s1,s2,s3]=size(hc);
        hc=reshape(hc,s1*s2,s3);
        hc=(hc-repmat(min(hc,[],2),[1,s3]))./repmat(max(hc,[],2)-min(hc,[],2),[1,s3]);
        hc=reshape(hc,s1,s2,s3);
    end
end