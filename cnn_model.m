classdef cnn_model < hypercube_ensemble
    % spectral-angle-mapper model using hypercubes / hypercube-ensembles
    
    properties
        classes                     % name of the reference classes
        nets                        % trained nets
        netinfo
        results                     % net resuts
        image_options
        augmentation_options
        net_options
        training_options
        class_id
        test_idx
    end
    
    methods
        function cnn = cnn_model(hc_ensemble, classes, options, exp_name, pth)
            %
            cnn.hypercube_list = hc_ensemble.hypercube_list;   
            cnn.filename = hc_ensemble.filename;              
            cnn.pc  = hc_ensemble.pc;                     
            cnn.mean_data = hc_ensemble.mean_data;              
            cnn.explained = hc_ensemble.explained;                   
            cnn.gTruth = hc_ensemble.gTruth;                    
            cnn.score_minmax = hc_ensemble.score_minmax; 
            cnn.history = hc_ensemble.history;                    
            cnn.function_queue = hc_ensemble.function_queue;
            
            cnn.image_options{1} = [];
            cnn.augmentation_options{1} = [];
            cnn.net_options{1} = [];
            cnn.training_options{1} = [];
            
            if nargin < 5 || isempty(pth); pth = cd; end
            if nargin < 4 || isempty(exp_name)
                time = clock; 
                exp_name = strcat('CNNet_',num2str(time(1)),'-',num2str(time(2)),...
                    '-',num2str(time(3)),'-',num2str(time(4)),'-',num2str(time(5)));
            end
            addpath(strcat(pth,'\',exp_name));
            cnn.filename = exp_name;
            if nargin == 3
                i = size(cnn.image_options,1);
                cnn.image_options{1} = image_opt(options);
                cnn.augmentation_options{1} = augment_opt(options);
                cnn.training_options{1} = cnnet_opt(options);
                cnn.net_options{1} = train_opt(options);
            end
            hc = load(cnn.hypercube_list{1});
            tmp = fieldnames(hc);
            hc = hc.(tmp{1}); 
            if nargin < 2
                classes = [];
            end
            if ~isempty(classes)
                classes_ = classes;
            elseif ~(isempty(hc.label_image))
                classes_ = categories(hc.label_image);
            elseif ~(isempty(cnn.gTruth))
                classes_ = cnn.gTruth;
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
            cnn.classes = classes_;
            cnn.class_id = class_id;
            
            % Erstellen von zusammengesetzten Klassennamen
            clear classes_
            for n=1:length(classes)
                temp=[];
                for l = 1:length(classes{n})
                    temp = strcat(temp,classes{n}(l));
                end
                classes_{n}=temp;
                if iscell(classes_{n})
                    classes_{n} = classes_{n}{1};
                end
            end
                
            
            for i = 1:length(cnn.hypercube_list)
                disp(i)
                load(cnn.hypercube_list{i});
                in = false;
                for m=1:length(classes)
                    idx = strcmp(char(hc.label),cnn.classes{m});
                    for l=1:length(idx)
                        if idx(l)
                            in = true;
                        end
                    end
                end
                if ~in
                    cnn.hypercube_list{i}=[];
                end
            end
            clear idx
            for i=1:numel(cnn.hypercube_list)
                if isempty(cnn.hypercube_list{i})
                    idx(i,1)=true;
                end
            end
            cnn.hypercube_list(idx)=[];
            
            cd(pth)
            mkdir(exp_name)
            cd(exp_name)
            cnn.pth = cd();
            save(exp_name,'cnn');
            addpath(strcat(pth,'\',exp_name));
        end 
        
        
        function self = train_model(self, model_num, options)
            if nargin < 3 && isempty(model_num) || nargin < 2
                model_num = size(self.results,2);
            end
            if nargin == 3 && isempty(model_num)
                i = size(self.results,2);
                self.image_options{i+1,1} = image_opt(options);
                self.augmentation_options{i+1,1} = augment_opt(options);
                self.training_options{i+1,1} = train_opt(options);
                self.net_options{i+1,1} = cnnet_opt(options);
            elseif nargin == 3 && ~(isempty(model_num))
                self.image_options{model_num,1} = image_opt(options);
                self.augmentation_options{model_num,1} = augment_opt(options);
                self.training_options{model_num,1} = train_opt(options);
                self.net_options{model_num,1} = cnnet_opt(options);                
            end
            if isempty(model_num)
                model_num = size(self.results,2) + 1;
            end
            imopts = self.image_options{model_num,1};
            augopts = self.augmentation_options{model_num,1};
            trainopts = self.training_options{model_num,1};
            netopts = self.net_options{model_num,1};            
                
            % create image augmenter
            augmenter = imageDataAugmenter('RandXReflection',augopts.xref,...
            'RandYReflection',augopts.yref,'RandYTranslation',augopts.xtrans,...
            'RandXTranslation',augopts.ytrans,'RandXScale',augopts.xscale,...
            'RandYScale',augopts.yscale,'RandXShear',augopts.xshear,...
            'RandYShear',augopts.yshear,'SetEDTransform',augopts.edtrans,...
            'RandGaussNoise',augopts.gauss,'RandRotation',augopts.rotation);         
            
            % create net
            [layers, imopts, self] = create_net(self, netopts, imopts, model_num);
            
                        % create training images
            [self, imds] = self.create_train_images(imopts, netopts);
            
            yt_ = []; yv_ = []; yT_ = [];
            for i = 1: trainopts.num_repeats
                % split data

                save('temp.mat','imds');
                [self, imdstrain, imdsval, imdstest] = split_data(self, imds, trainopts, i);
                load('temp.mat');
                
                % image normalization
                [imdstrain, imdsval, imdstest] = image_normalization(imdstrain, imdsval, imdstest, imopts, self.pth);
                
                % balance training data
                %imdstrain = databalancer(imdstrain);
                
                yt_ = vertcat(yt_, imdstrain.Labels);
                if ~(isempty(imdsval))
                    yv_ = vertcat(yv_, imdsval.Labels);
                end
                if ~isempty(imdstest)
                    yT_ = vertcat(yT_, imdstest.Labels);
                end
                
                if strcmp(netopts.mode, 'classification')
                    traindata = augmentedImageSource([imopts.image_size(1) imopts.image_size(2) length(imopts.data_range)],...
                        imdstrain, 'DataAugmentation', augmenter, 'BackgroundExecution', augopts.bg_excec);
                    if ~(isempty(imdsval))
                        valdata = imdsval;
                    else
                        valdata = [];
                    end
                elseif strcmp(netopts.mode, 'regression')
                    traindata = table(imdstrain.Files, imdstrain.Labels);
                    %traindata = augmentedImageSource([imopts.image_size(1) imopts.image_size(2) length(imopts.data_range)],...
                    %    imdstrain, 'DataAugmentation', augmenter, 'BackgroundExecution', augopts.bg_excec);
                    if ~(isempty(imdsval))
                        valdata = table(imdsval.Files, imdsval.Labels);
                    else
                        valdata = [];
                    end
                end
                
                % create net_options

                valfreq = floor(numel(imdstrain.Files)/trainopts.bs)*trainopts.valfreq;

                options=trainingOptions('sgdm','Momentum',trainopts.momentum,'InitialLearnRate',trainopts.lr,...
                'L2Regularization',trainopts.l2_reg,'MaxEpochs',trainopts.maxepochs,'MiniBatchSize',trainopts.bs,...
                'Shuffle',trainopts.shuffle,'Plots',trainopts.plots,'LearnRateSchedule',trainopts.lr_schedule,...
                'LearnRateDropFactor',trainopts.lr_drop_factor,'ValidationData',...
                valdata, 'LearnRateDropPeriod',trainopts.lr_drop_period,'Verbose',trainopts.verbose,...
                'ValidationFrequency',valfreq,'ValidationPatience',Inf,'OutputFcn',...
                @(info)stopfun(info,trainopts.valpatience,netopts.mode),'ExecutionEnvironment','auto'); %,'OutputFcn',@(info)stopfun(info,5)  % ,'ExecutionEnvironment','gpu'

                % train net
                tic
                disp(strcat('Modell_',num2str(model_num),'   Versuch: ', num2str(i)))
                
                try
                    error = 0;
                    [self.nets{model_num,i}, self.netinfo{model_num,i}] = trainNetwork(traindata, layers, options);
                catch
                    warning('Problem training net. Continuing to next net.')
                    self.nets{model_num,i} = [];
                    self.netinfo{model_num,i} = [];
                    error = 1;
                    continue
                end
                disp('benötigte Zeit:');
                self.results(model_num).time(i) = toc;
                disp(self.results(model_num).time(i));
                fprintf('\n')
                
                % test net
                cd(self.pth);
                mkdir(strcat('Modell_',num2str(model_num)));
                cd(strcat('Modell_',num2str(model_num)));
                mkdir(strcat('Versuch_',num2str(i)));
                cd(strcat('Versuch_',num2str(i)));
                
                % results
                if strcmp(netopts.mode, 'classification')
                    yt{i} = classify(self.nets{model_num,i}, imdstrain);
                    %draw_confusionmat(imdstrain.Labels, yt(:,i), self.classes, 'training');
                    self.results(model_num).train_cm{i} = confusionmat(imdstrain.Labels,yt{i});
                    self.results(model_num).train(i) = BM(self.results(model_num).train_cm{i});

                    if ~(isempty(imdsval))
                        yv{i} = classify(self.nets{model_num,i}, imdsval);
                        %draw_confusionmat(imdsval.Labels, yv(:,i), self.classes, 'validation');
                        self.results(model_num).val_cm{i} = confusionmat(imdsval.Labels,yv{i});
                        self.results(model_num).validation(i) = BM(self.results(model_num).val_cm{i});
                    end
                    if ~(isempty(imdstest))
                        yT{i} = classify(self.nets{model_num,i}, imdstest);
                        %draw_confusionmat(imdstest.Labels, yT(:,i), self.classes, 'test');
                        self.results(model_num).test_cm{i} = confusionmat(imdstest.Labels,yT{i});
                        self.results(model_num).test(i) = BM(self.results(model_num).test_cm{i});
                    end
                elseif strcmp(netopts.mode, 'regression')
                    yt{i} = predict(self.nets{model_num,i}, imdstrain);
                    self.results(model_num).predError_train{i} = imdstrain.Labels - yt{i};
                    self.results(model_num).MSE_train(i) = mean((imdstrain.Labels - yt{i}).^2);
                    self.results(model_num).RMSE_train(i) = sqrt(mean((imdstrain.Labels - yt{i}).^2));
                    self.results(model_num).r2_train(i) = draw_regression(imdstrain.Labels, yt{i}, 'training')
                    if ~(isempty(imdsval))
                        yv{i} = predict(self.nets{model_num,i}, imdsval);
                        self.results(model_num).predError_val{i} = imdsval.Labels - yv{i};
                        self.results(model_num).MSE_val(i) = mean((imdsval.Labels - yv{i}).^2);
                        self.results(model_num).RMSE_val(i) = sqrt(mean((imdsval.Labels - yv{i}).^2));
                        self.results(model_num).r2_val(i) = draw_regression(imdsval.Labels, yv{i}, 'validation')
                    end
                    if ~(isempty(imdstest))
                        yT{i} = predict(self.nets{model_num,i}, imdstest);
                        self.results(model_num).predError_test{i} = imdstest.Labels - yT{i};
                        self.results(model_num).MSE_test(i) = mean((imdstest.Labels - yT{i}).^2);
                        self.results(model_num).RMSE_test(i) = sqrt(mean((imdstest.Labels - yT{i}).^2));
                        self.results(model_num).r2_test(i) = draw_regression(imdstest.Labels, yT{i}, 'test');
                    end
                end
                % save net if save_net = true
                if netopts.save_net
                    cnn = self.nets{model_num,i};
                    save('CNN-Net','cnn');
                end
                self.nets{model_num,i} = [];
                reset(gpuDevice);
            end
            
            cd('..')
                     
            if error
                cd(self.pth);
                cnn = self;
                save(self.filename,'cnn');
                return
            end
                
            if strcmp(netopts.mode, 'classification')
                y=[];for i=1:size(yt,2);y=vertcat(y,yt{i});end
                yt = y;
                draw_confusionmat(yt_, yt, self.classes, 'training_mean');
                if ~(isempty(imdsval))
                    y=[];for i=1:size(yv,2);y=vertcat(y,yv{i});end
                    yv = y;
                    draw_confusionmat(yv_, yv, self.classes, 'validation_mean');
                end
                if ~(isempty(imdstest))
                    y=[];for i=1:size(yT,2);y=vertcat(y,yT{i});end
                    yT = y;
                    draw_confusionmat(yT_, yT, self.classes, 'test_mean');
                end
            elseif strcmp(netopts.mode, 'regression')
                y=[];for i=1:size(yt,2);y=vertcat(y,yt{i});end
                yt = y;
                draw_regression(yt_, yt, 'training_mean')
                if ~(isempty(imdsval))
                    y=[];for i=1:size(yv,2);y=vertcat(y,yv{i});end
                    yv = y;
                    draw_regression(yv_, yv, 'validation_mean')
                end
                if ~(isempty(imdstest))
                    y=[];for i=1:size(yT,2);y=vertcat(y,yT{i});end
                    yT = y;
                    draw_regression(yT_, yT, 'test_mean')
                end
            end
            
            
            % save cnn model
            cd(self.pth);
            cnn = self;
            save(self.filename,'cnn');
        end
        function [self, imds] = create_train_images(self, imopts, netopts)
            if strcmp(imopts.data_range, ':')
                range = 'all';
            else
                range = num2str(imopts.data_range);
            end
            image_folder = strcat('Images_',imopts.image_data,...
                '_',strrep(range,'  ','-'),'_',...
                num2str(imopts.image_size(1)), 'x',...
                num2str(imopts.image_size(2)),'_',imopts.resize_method);
            cd(self.pth)
            tempname = strcat(self.pth,'\',image_folder);
            if exist(tempname,'dir') ~= 7
                mkdir(tempname);
                create_images(self,imopts,tempname);
                cd(tempname);
                cd('Labels')
                load('Labels.mat','labels');
            else
                cd(tempname);
                cd('Labels')
                load('Labels.mat','labels');              
            end

            cd(self.pth);
            imds = imageDatastore(tempname,'FileExtensions','.mat');
            
%             files = cell(length(self.hypercube_list),1);
%             for i=1:length(self.hypercube_list)
%                 [~,name,~] = fileparts(self.hypercube_list{i});
%                 files{i} = strcat(tempname,'\',name,'_Image.mat');
%             end
%             imds.Files = files;
            
            imds.ReadFcn = @matimreader;
            if strcmp(netopts.mode, 'classification')
                imds.Labels = categorical(labels);
            elseif strcmp(netopts.mode, 'regression')
                imds.Labels = str2double(labels);
            end
        end
    end
end

%--------------------------------------------------------------------------
% supporting functions
function [imdstrain, imdsval, imdstest] = image_normalization(imdstrain, imdsval, imdstest, options, pth)

if strcmp(options.normalization, 'none')
    return
end

pth_org = cd();
cd(pth)
mkdir('Temp')
cd('Temp')
delete *.mat
cd('..')

for i = 1:length(imdstrain.Files)
    load(imdstrain.Files{i});
    %fprintf('Image %i of %i read \n',i, length(imdstrain.Files))
    if i == 1
    	meanim = zeros(size(im));
    	stdim = zeros(size(im));
    end
    
    [im] = normalisator(im, options, [], []);
    
    meanim = meanim + im;
    
    [~,name,~] = fileparts(imdstrain.Files{i});
    savepth = strcat(pth,'\Temp\',name,'.mat');
    save(savepth,'im');
    imdstrain.Files{i} = savepth;
end



if strcmp(options.normalization, 'mean3') || strcmp(options.normalization, 'norm3')
    meanim = meanim ./ length(imdstrain.Files);
%     for i=1:size(meanim,3)
%         temp = meanim(:,:,i);
%         meanim_(i) = mean(temp(:));
%     end
    for i = 1:length(imdstrain.Files)
        load(imdstrain.Files{i});
        %fprintf('Image %i of %i read \n',i, length(imdstrain.Files))
        for l = 1:size(im,3)
            im(:,:,l) = im(:,:,l) - meanim(:,:,l);
            %stdim(l) = stdim(l) + (sum(sum(im(:,:,l).^2))/(size(im,1)*size(im,2)));
            stdim(:,:,l) = stdim(:,:,l) + (((im(:,:,l).^2))/(size(im,1)*size(im,2)));
        end
        save(imdstrain.Files{i},'im');
    end
    mean_ = meanim;
else
    mean_=[];
    std_=[];
end
if strcmp(options.normalization, 'norm3')
    stdim = sqrt(stdim ./ length(imdstrain.Files));
    for i = 1:length(imdstrain.Files)
        load(imdstrain.Files{i});
        %fprintf('Image %i of %i read \n',i, length(imdstrain.Files))
        for l = 1:size(im,3)
            im(:,:,l) = im(:,:,l) ./ stdim(:,:,l);
        end
        save(imdstrain.Files{i},'im');
    end
     std_ = stdim;
else
    std_ = [];
end         

if ~isempty(imdsval)
    for i = 1:length(imdsval.Files)
        load(imdsval.Files{i})
        %fprintf('Image %i of %i read \n',i, length(imdsval.Files))
        im = normalisator(im, options, mean_, std_);
        [~,name,~] = fileparts(imdsval.Files{i});
        savepth = strcat(pth,'\Temp\',name,'.mat');
        save(savepth,'im');
        imdsval.Files{i} = savepth;        
    end
end
if ~isempty(imdstest)
    for i = 1:length(imdstest.Files)
        load(imdstest.Files{i})
        %fprintf('Image %i of %i read \n',i, length(imdstest.Files))
        im = normalisator(im, options, mean_, std_);
        [~,name,~] = fileparts(imdstest.Files{i});
        savepth = strcat(pth,'\Temp\',name,'.mat');
        save(savepth,'im');
        imdstest.Files{i} = savepth;         
    end
end
end

function [im] = normalisator(im, options, mean_, std_)
    
if strcmp(options.normalization, 'gcn')
    x = im;
    xmean = mean(mean(mean(x)));
    x = x-xmean;
    con = sqrt(1+mean(x(:).^2));
    im=1*x./max([con,0.0000001]);
elseif strcmp(options.normalization, 'mean1')
    im = im - mean(im(:));
elseif strcmp(options.normalization, 'mean2')
    for l = 1: size(im,3)
        temp = im(:,:,l);
        im(:,:,l) = im(:,:,l) - mean(temp(:));
    end
    clear temp
elseif strcmp(options.normalization, 'norm1')
    im = im - mean(im(:));
    im = im ./ std(im(:));
elseif strcmp(options.normalization, 'norm2')
    for l = 1: size(im,3)
        temp = im(:,:,l);
        im(:,:,l) = im(:,:,l) - mean(temp(:));
    end
    clear temp
    for l = 1: size(im,3)
        temp = im(:,:,l);
        im(:,:,l) = im(:,:,l) ./ std(temp(:));
    end
    clear temp
elseif strcmp(options.normalization, 'mean3') && ~isempty(mean_)
    for i = 1:size(im,3)
        im(:,:,i) = im(:,:,i) - mean_(:,:,i);
    end    
elseif strcmp(options.normalization, 'norm3')  && ~isempty(std_)
    for i = 1:size(im,3)
        im(:,:,i) = (im(:,:,i) - mean_(:,:,i))./std_(:,:,i);
    end   
elseif strcmp(options.normalization, 'none')
else
    return
end
    
end
    
function create_images(seg_model, options, pth)

n = 1;
labels = cell(1,size(seg_model.hypercube_list,1));

classes_={};
for i=1:length(seg_model.classes)
    temp = [];
    if iscell(seg_model.classes{i})
        for l=1:length(seg_model.classes{i})
            temp = strcat(temp,seg_model.classes{i}(l));
        end
        classes_{i}=temp{1};
    else
        classes_{i}=seg_model.classes{i};
    end
end
        
for i = 1:numel(seg_model.hypercube_list)
    hc = load(seg_model.hypercube_list{i});

    %fprintf('Hypercube %i of %i read \n',i, numel(seg_model.hypercube_list))
    
    tmp = fieldnames(hc);
    hc = hc.(tmp{1});
    
    value = 0;
    for l = 1:length(seg_model.classes)
        idx = strcmp(char(hc.label),seg_model.classes{l});
        for m=1:length(idx)
            if idx(m)
                value = 1;
                classidx = l;
            end
        end
    end
    
    if value == 1
        labels{n} = classes_{classidx};
        %labels{n} = char(hc.label);
        im = hc.(options.image_data)(:,:,options.data_range);
        if islogical(im)
            im = double(im);
        end
%         %%%%%%%%%%%%%%%%%%%%%%%%%%%
%         im = mat2gray(im);
%         %%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        if size(im,3) == 2
            im(:,:,3) = zeros(size(im(:,:,1)));
        elseif size(im,3) > 3
            im = im(:,:,1:3);
        end

        if options.image_size(1) ~= size(im,1) || options.image_size(2) ~= size(im,2)
            if strcmp(options.resize_method,'resize')
                im = resizeimages(im,options.image_size(1),options.image_size(2),0);
            elseif strcmp(options.resize_method,'padding')
                im = resizeimages(im,options.image_size(1),options.image_size(2),1);
            elseif strcmp(options.resize_method,'none')
            else
                error('No valid resize method choosen. Choose "resize", "padding" or "none".');
            end
        end
        save(strcat(pth, '\', hc.filename,'_Image.mat'), 'im');
        n = n+1;
    end
end

labels = labels(~cellfun('isempty',labels));

cd(pth)
mkdir('Labels')
cd('Labels')
save('Labels.mat','labels');

end

function [resized_im] = resizeimages(im,row,col,scale)

% Resizes each image in the image array images (e.g. from mia_blobimages)
% to the size defined by row and col.

% scale ... If 0 images are resized using the imresize function. If 1
% images are cropped ore zero-padded to the desired size.

% Outputs a 3d array ... [row,col,s3]

% if mod(row,2) ~= 0 && row ~= 1
%     row = row + 1;
% end
% if mod(col,2) ~= 0
%     col = col + 1;
% end

if scale == 0
    resized_im=imresize(im,[row,col],'nearest');
elseif scale == 1
    [s1, s2, ~] = size(im);
    while s1 < row
        im = padarray(im,[1 0]);
        s1 = s1 + 2;
    end
    while s2 < col
        im = padarray(im,[0 1]);
        s2 = s2 + 2;
    end
    if s1 > row || s2 > col
        [c1, c2, ~] = size(im);
        c1 = ceil(c1/2); c2 = ceil(c2/2);
        im = im(c1-row/2+1:c1-row/2+row, c2-col/2+1:c2-col/2+col, :);
    end
 resized_im = im; 
elseif scale == 2
    [s1, s2, ~] = size(im);
    while s1 < row
        temp = categorical(ones(1, s2)*nan);
        im = vertcat(temp, im, temp);
        s1 = s1 + 2;
    end
    while s2 < col
        temp = categorical(ones(s1, 1)*nan);
        im = horzcat(temp, im, temp);
        s2 = s2 + 2;
    end
    if s1 > row || s2 > col
        [c1, c2, ~] = size(im);
        c1 = ceil(c1/2); c2 = ceil(c2/2);
        im = im(c1-row/2+1:c1-row/2+row, c2-col/2+1:c2-col/2+col, :);
    end
 resized_im = im;
end
end

function [layers, imoptions, cnn_model] = create_net(cnn_model, netoptions, imoptions, model_num)

if isempty(netoptions.prenet)
    if strcmp(netoptions.mode, 'classification')
     inputlayer = imageInputLayer([imoptions.image_size(1) imoptions.image_size(2) length(imoptions.data_range)],...
         'Normalization','none');
    elseif strcmp(netoptions.mode, 'regression')
        inputlayer = imageInputLayer([imoptions.image_size(1) imoptions.image_size(2) length(imoptions.data_range)],...
         'Normalization','none','DataAugmentation','randfliplr');
    end

     % batchnormalization  
     if netoptions.bn
        bnlayer = batchNormalizationLayer;
    else
        bnlayer = [];
     end
    
    % relus
    if strcmp(netoptions.relu, 'relu')
        relulay = reluLayer;
    elseif strcmp(netoptions.relu, 'lrelu')
        relulay = leakyReluLayer(parameter.reLu_leaky);
    elseif strcmp(netoptions.relu, 'crelu')
        relulay = clippedReluLayer(parameter.reLu_clipping);
    elseif strcmp(netoptions.relu, 'elu')
        relulay = [];
        bnlayer = [];
    end

    num1 = netoptions.layers + netoptions.fc;
    num2 = netoptions.layers * netoptions.depth;
    num3 = netoptions.layers;

    %fc
    if length(netoptions.fc_size) < netoptions.fc
        netoptions.fc_size(length(netoptions.fc_size)+1:netoptions.fc) = netoptions.fc_size(1);
    elseif length(netoptions.fc_size) > netoptions.fc
        netoptions.fc_size = netoptions.fc_size(1:netoptions.fc);
    end
    
    %dropout
    if length(netoptions.do) < num1
        netoptions.do(length(netoptions.do)+1:num1) = netoptions.do(1);
    elseif length(netoptions.do) > num1
        netoptions.do = netoptions.do(1:num1);
    end
    
    %size convolution
    if size(netoptions.conv_size,1) < num2
            netoptions.conv_size(size(netoptions.conv_size,1)+1:num2,:) = repmat(netoptions.conv_size(1,:),[length(size(netoptions.conv_size,1)+1:num2) 1]);
    elseif size(netoptions.conv_size,1) > num2
            netoptions.conv_size = netoptions.conv_size(1:num2,:);
    end
    
    %stride convolution
    if size(netoptions.conv_stride,1) < num2
            netoptions.conv_stride(size(netoptions.conv_stride,1)+1:num2, :) = repmat(netoptions.conv_stride(1,:),[length(size(netoptions.conv_stride,1)+1:num2) 1]);
    elseif size(netoptions.conv_stride,1) > num2
            netoptions.conv_stride = netoptions.conv_stride(1:num2,:);
    end
    
    %size pooling
    if size(netoptions.pool_size,1) < num3
            netoptions.pool_size(size(netoptions.pool_size,1)+1:num3, :) = repmat(netoptions.pool_size(1,:),[length(size(netoptions.pool_size,1)+1:num3) 1]);
    elseif size(netoptions.pool_size,1) > num3
            netoptions.pool_size = netoptions.pool_size(1:num3,:);
    end
    
    %stride pooling
    if size(netoptions.pool_stride,1) < num3
            netoptions.pool_stride(size(netoptions.pool_stride,1)+1:num3, :) = repmat(netoptions.pool_stride(1,:),[length(size(netoptions.pool_stride,1)+1:num3) 1]);
    elseif size(netoptions.pool_stride,1) > num3
            netoptions.pool_stride = netoptions.pool_stride(1:num3,:);
    end
    
    % number of filters
    if strcmp(netoptions.filter, 'fix')
        netoptions.num_filter(1: num2) = netoptions.num_filter;
    elseif strcmp(netoptions.filter, 'double')
        f = [];
        for i = 1:num3
            f = vertcat(f,ones(netoptions.depth,1) * netoptions.num_filter);
            netoptions.num_filter = netoptions.num_filter * 2;
        end
        netoptions.num_filter = f;
    end
    
    % build network
    layers = inputlayer;
    m = 1;
    for i = 1: netoptions.layers
        % do
        if netoptions.do(i) ~= 0
            layers = [layers, dropoutLayer(netoptions.do(i))];
        end
        for n = 1: netoptions.depth
            layers = [layers, convolution2dLayer(netoptions.conv_size(m,:),...
                netoptions.num_filter(m),'Padding','same','Stride',netoptions.conv_stride(m,:))];
            layers = [layers, relulay, bnlayer];
            if strcmp(netoptions.relu, 'elu')
                layers = [layers, eluLayer(netoptions.num_filter(m))];
            end
            m = m+1;
        end
        if strcmp(netoptions.pool, 'average')
            layers = [layers, averagePooling2dLayer(netoptions.pool_size(i,:),...
                'Stride',netoptions.pool_stride(i,:))];
        else
            layers = [layers, maxPooling2dLayer(netoptions.pool_size(i,:),...
                'Stride',netoptions.pool_stride(i,:))];        
        end
    end
    for i = 1: netoptions.fc
        if netoptions.do(netoptions.layers + 1) ~= 0
            layers = [layers,  dropoutLayer(netoptions.do(netoptions.layers + 1))];
        end
        layers = [layers, fullyConnectedLayer(netoptions.fc_size(i))];
        layers = [layers, relulay, bnlayer];
        if strcmp(netoptions.relu, 'elu')
            layers = [layers, eluLayer(netoptions.fc_size(i))];
        end
    end
    if strcmp(netoptions.mode, 'classification')
        layers = [layers, fullyConnectedLayer(length(cnn_model.classes)), softmaxLayer, classificationLayer()];
    elseif strcmp(netoptions.mode, 'regression')
        layers = [layers, fullyConnectedLayer(1), regressionLayer()];
    end

elseif isnumeric(netoptions.prenet)
    if netoptions.prenet > 0
        if size(cnn_model.nets,1) > netoptions.prenet
            n = netoptions.prenet;
        else
            error('This net was not trained.');
        end
        if size([cnn_model.nets{n,:}],2) > 1
            [~, m] = max([cnn_model.results(n).MeanAcc_train]);
        else
            m = 1;
        end
    elseif netoptions.prenet == (-1)
        if size(cnn_model.nets,1) > 0
            n = size(cnn_model.nets,1) - 1;
        else
            error('No net was trained.');
        end
        if size([cnn_model.nets{n,:}],2) > 1
            [~, m] = max([cnn_model.results(n).MeanAcc_train]);
        else
            m = 1;
        end
    else
        error('Incorrect pre-trained net. Choose "-1" for the previous net or any number of an already trained net');
    end
        net = cnn_model.nets{n,m};
        for i = 1:size(net.Layers,1)
            if strcmp(net.Layers(i).Name,'imageInput')
                laynum = i;
            end
        end
        
        imsize = net.Layers(laynum).InputSize;
        if isa(net, 'DAGNetwork')
            lgraph = layerGraph(net);
            lgraph=removeLayers(lgraph,'labels');
            claslay=classificationLayer('Name','labels','ClassNames',cnn_model.classes);
            lgraph=addLayers(lgraph,claslay);
            layers=connectLayers(lgraph,'softmax','labels');
        elseif isa(net, 'SeriesNetwork')
            laytrans = net.Layers(1:end-3);
            if strcmp(netoptions.mode, 'classification')
                layers = [laytrans, fullyConnectedLayer(length(cnn_model.classes)), softmaxLayer, classificationLayer];               
            elseif strcmp(netoptions.mode, 'regression')
                layers = [laytrans, fullyConnectedLayer(1), regressionLayer];
            end
        else
           error('Wrong Net loaded.'); 
        end
        
        if imoptions.image_size(1) ~= imsize(1) || imoptions.image_size(2) ~= imsize(2)
            imoptions.image_size = imsize;
            cnn_model.image_options{model_num,1}.image_size = imsize;            
            fprintf('Image Size was changed to %i x %i to match size of the pre-trained net.\n',imsize(1),imsize(2));
        end
        if length(imoptions.data_range) > imsize(3)
            imoptions.data_range = imoptions.data_range(1:imsize(3));
            cnn_model.image_options{model_num,1}.data_range = imoptions.data_range;
            fprintf('Data Range was changed to  %i to match size of the pre-trained net.\n',imsize(3));
        elseif length(imoptions.data_range) == imsize(3)
        else
            error('Not enough image channels to use the pre-trained net');
        end        
        
elseif strcmp(netoptions.prenet,'vgg16') || strcmp(netoptions.prenet,'vgg19')...
        || strcmp(netoptions.prenet,'googlenet') || strcmp(netoptions.prenet,'alexnet')
     net = feval(netoptions.prenet);
     if isa(net,'SeriesNetwork')
        layersTransfer = net.Layers(1:end-3);
        numClasses = numel(cnn_model.classes);
        layers = [
        layersTransfer
        fullyConnectedLayer(numClasses)
        softmaxLayer
        classificationLayer];
        imoptions.image_size=layers(1, 1).InputSize;
        inputlayer = imageInputLayer(layers(1, 1).InputSize,...
         'Normalization','none');
        layers = layers(2:end);
        layers = [inputlayer; layers];
     else
        lgraph = layerGraph(net);
        inputSize = net.Layers(1).InputSize;
        for l=1:3
            laynames{l} = lgraph.Layers(end-l+1,1).Name;
        end
        lgraph = removeLayers(lgraph, laynames);
        newLayers = [
            fullyConnectedLayer(numel(cnn_model.classes),'Name','fc')
            softmaxLayer('Name','softmax')
            classificationLayer('Name','classoutput')];
        lgraph = addLayers(lgraph,newLayers);
        lgraph = connectLayers(lgraph,lgraph.Layers(end-3,1).Name,'fc'); 
        imoptions.image_size=lgraph.Layers(1, 1).InputSize;
        inputlayer = imageInputLayer(imoptions.image_size,...
         'Normalization','none','Name','inpLay');
        lgraph = removeLayers(lgraph, lgraph.Layers(1,1).Name);
        lgraph = addLayers(lgraph,inputlayer);
        layers = connectLayers(lgraph,'inpLay',lgraph.Layers(1,1).Name);
     end
        %error('Not implemented.');
elseif strcmp(netoptions.prenet,'load') 
    orgpth = cd;
    [filename, pathname]=uigetfile('*','Dateien Wählen','Multiselect','Off');
    cd(pathname);
    net = load(filename);
    fprintf('Pre-trained net loaded.')
    tmp = fieldnames(net);
    net = net.(tmp{1});
    for i = 1:size(net.Layers,1)
        if strcmp(net.Layers(i).Name,'imageInput')
            laynum = i;
        end
    end

    imsize = net.Layers(laynum).InputSize;
        if isa(net, 'DAGNetwork')
            lgraph = layerGraph(net);
            lgraph=removeLayers(lgraph,'labels');
            claslay=classificationLayer('Name','labels','ClassNames',cnn_model.classes);
            lgraph=addLayers(lgraph,claslay);
            layers=connectLayers(lgraph,'softmax','labels');
        elseif isa(net, 'SeriesNetwork')
            laytrans = net.Layers(1:end-3);
            if strcmp(netoptions.mode, 'classification')
                layers = [
                    laytrans
                    fullyConnectedLayer(length(cnn_model.classes))
                    softmaxLayer
                    classificationLayer];                
            elseif strcmp(netoptions.mode, 'regression')
                layers = [
                    laytrans
                    fullyConnectedLayer(1)
                    regressionLayer];
            end
        else
           error('Wrong Net loaded.'); 
        end

        if imoptions.image_size(1) ~= imsize(1) || imoptions.image_size(2) ~= imsize(2)
            imoptions.image_size = imsize;
            cnn_model.image_options{model_num,1}.image_size = imsize;            
            fprintf('Image Size was changed to %i x %i to match size of the pre-trained net.\n',imsize(1),imsize(2));
        end
        if length(imoptions.data_range) > imsize(3)
            imoptions.data_range = imoptions.data_range(1:imsize(3));
            cnn_model.image_options{model_num,1}.data_range = imoptions.data_range;
            fprintf('Data Range was changed to  %i to match size of the pre-trained net.\n',imsize(3));
        elseif length(imoptions.data_range) == imsize(3)
        else
            error('Not enough image channels to use the pre-trained net');
        end        
        cd(orgpth);
% elseif strcmp(netoptions.prenet(1:4),'load') 
%     orgpth = cd;
%     filename=netoptions.prenet(6:end);
%     net = load(filename);
%     fprintf('Pre-trained net loaded.')
%     tmp = fieldnames(net);
%     net = net.(tmp{1});
%     for i = 1:size(net.Layers,1)
%         if strcmp(net.Layers(i).Name,'imageinput')
%             laynum = i;
%         end
%     end
% 
%     imsize = net.Layers(laynum).InputSize;
%         if isa(net, 'DAGNetwork')
%             lgraph = layerGraph(net);
%             lgraph=removeLayers(lgraph,'labels');
%             claslay=classificationLayer('Name','labels','ClassNames',cnn_model.classes);
%             lgraph=addLayers(lgraph,claslay);
%             layers=connectLayers(lgraph,'softmax','labels');
%         elseif isa(net, 'SeriesNetwork')
%             laytrans = net.Layers(1:end-3);
%             if strcmp(netoptions.mode, 'classification')
%                 layers = [
%                     laytrans
%                     fullyConnectedLayer(length(cnn_model.classes))
%                     softmaxLayer
%                     classificationLayer];                
%             elseif strcmp(netoptions.mode, 'regression')
%                 layers = [
%                     laytrans
%                     fullyConnectedLayer(1)
%                     regressionLayer];
%             end
%         else
%            error('Wrong Net loaded.'); 
%         end
% 
%         if imoptions.image_size(1) ~= imsize(1) || imoptions.image_size(2) ~= imsize(2)
%             imoptions.image_size = imsize;
%             cnn_model.image_options{model_num,1}.image_size = imsize;            
%             fprintf('Image Size was changed to %i x %i to match size of the pre-trained net.\n',imsize(1),imsize(2));
%         end
%         if length(imoptions.data_range) > imsize(3)
%             imoptions.data_range = imoptions.data_range(1:imsize(3));
%             cnn_model.image_options{model_num,1}.data_range = imoptions.data_range;
%             fprintf('Data Range was changed to  %i to match size of the pre-trained net.\n',imsize(3));
%         elseif length(imoptions.data_range) == imsize(3)
%         else
%             error('Not enough image channels to use the pre-trained net');
%         end        
%         cd(orgpth);
elseif isa(netoptions.prenet,'DAGNetwork') || isa(netoptions.prenet,'SeriesNetwork') 
    net = netoptions.prenet;
    %tmp = fieldnames(net);
    %net = net.(tmp{1});
%     for i = 1:size(net.Layers,1)
%         if isfield(net.Layers(i,1),'Name') && strcmp(net.Layers(i).Name,'imageinput')
%             laynum = i;
%         end
%     end
    laynum = 1;
    imsize = net.Layers(laynum).InputSize;
        if isa(net, 'DAGNetwork')
            lgraph = layerGraph(net);
            lgraph=removeLayers(lgraph,'labels');
            claslay=classificationLayer('Name','labels','ClassNames',cnn_model.classes);
            lgraph=addLayers(lgraph,claslay);
            layers=connectLayers(lgraph,'softmax','labels');
        elseif isa(net, 'SeriesNetwork')
            laytrans = net.Layers(1:end-3);
            if strcmp(netoptions.mode, 'classification')
                layers = [
                    laytrans
                    fullyConnectedLayer(length(cnn_model.classes))
                    softmaxLayer
                    classificationLayer];                
            elseif strcmp(netoptions.mode, 'regression')
                layers = [
                    laytrans
                    fullyConnectedLayer(1)
                    regressionLayer];
            end
        else
           error('Wrong Net loaded.'); 
        end

        if imoptions.image_size(1) ~= imsize(1) || imoptions.image_size(2) ~= imsize(2)
            imoptions.image_size = imsize;
            cnn_model.image_options{model_num,1}.image_size = imsize;            
            fprintf('Image Size was changed to %i x %i to match size of the pre-trained net.\n',imsize(1),imsize(2));
        end
        if length(imoptions.data_range) > imsize(3)
            imoptions.data_range = imoptions.data_range(1:imsize(3));
            cnn_model.image_options{model_num,1}.data_range = imoptions.data_range;
            fprintf('Data Range was changed to  %i to match size of the pre-trained net.\n',imsize(3));
        elseif length(imoptions.data_range) == imsize(3)
        else
            error('Not enough image channels to use the pre-trained net');
        end        
else
    error('Error.');
end
end

function [cnnmdl, imdstrain, imdsval, imdstest] = split_data(cnnmdl, imds, options, num)
idx = cell(3,1);
if length(options.data_split{1}) ~= 1
    imds_=imds;
    testimages = imds.Files(logical(options.data_split{1}));
    testlabels = imds.Labels(logical(options.data_split{1}));
    imds.Files(logical(options.data_split{1})) = [];

    valimages = imds.Files(logical(options.data_split{2} == num));
    vallabels = imds.Labels(logical(options.data_split{2} == num));
    
    trainimages = imds.Files(logical(options.data_split{2} ~= num));
    trainlabels = imds.Labels(logical(options.data_split{2} ~= num));

    imdstrain = imageDatastore(trainimages, 'FileExtension','.mat');
    imdstrain.ReadFcn = @matimreader;
    imdstrain.Labels = trainlabels;
    if ~(isempty(valimages))
        imdsval = imageDatastore(valimages, 'FileExtension','.mat');
        imdsval.ReadFcn = @matimreader;
        imdsval.Labels = vallabels;
    else
        imdsval = [];
    end
    if ~(isempty(testimages))
        imdstest = imageDatastore(testimages, 'FileExtension','.mat');
        imdstest.ReadFcn = @matimreader;
        imdstest.Labels = testlabels;
    else
        imdstest = [];
    end
    imds=imds_;
else
    if ~isempty(cnnmdl.test_idx)
        options.data_split{3} = options.data_split(1:2);
        idx{3} = cnnmdl.test_idx;
    end
    % check data_split

    for i = 1:length(options.data_split)
        if ~isnumeric(options.data_split{i})
            data_split(i) = 0;
            for n = 1:length(options.data_split{i})
                for l = 1:size(imds.Files,1)
                    [~,name,~] = fileparts(imds.Files{l});
                    if contains(name,options.data_split{i}{n})
                        idx{i,1} = vertcat(idx{i,1}, l);
                    end
                end
            end            
        else
            data_split(i) = options.data_split(i);
        end
    end

    for i = 1:length(data_split)
        data_split_(i) = data_split{i}/sum([data_split{:}]);
    end
    data_split = data_split_;

    numfiles = numel(imds.Files);
    shuffleidx = 1:numfiles;

    for n = 1:3
        idxcommon = intersect(idx{n}, shuffleidx);
        shuffleidx = setxor(shuffleidx, idxcommon);
    end
    shuffleidx = shuffleidx(randperm(length(shuffleidx)));
    numfiles = length(shuffleidx);

    for i=1:3
        if isempty(idx{i})
            n = floor(data_split(i)*numfiles);
            idx{i} = shuffleidx(1:n);
            shuffleidx = shuffleidx(n+1:end);
        end
    end

    % image datastores
    trainimages = imds.Files(idx{1});
    trainlabels = imds.Labels(idx{1});
    valimages = imds.Files(idx{2});
    vallabels = imds.Labels(idx{2});
    testimages = imds.Files(idx{3});
    testlabels = imds.Labels(idx{3});
    imdstrain = imageDatastore(trainimages, 'FileExtension','.mat');
    imdstrain.ReadFcn = @matimreader;
    imdstrain.Labels = trainlabels;
    if ~(isempty(idx{2}))
        imdsval = imageDatastore(valimages, 'FileExtension','.mat');
        imdsval.ReadFcn = @matimreader;
        imdsval.Labels = vallabels;
    else
        imdsval = [];
    end
    if ~(isempty(idx{3}))
        imdstest = imageDatastore(testimages, 'FileExtension','.mat');
        imdstest.ReadFcn = @matimreader;
        imdstest.Labels = testlabels;
    else
        imdstest = [];
    end

    if isempty(cnnmdl.test_idx)
        cnnmdl.test_idx = idx{3};
    end
end
end

function imdstrain = databalancer(imdstrain)

%make training dataset balanced
unique_labels = unique(imdstrain.Labels);
max_labels = 0; maxnum_labels = 0;
for i = 1:length(unique_labels)
    idx_labels{i} = find(imdstrain.Labels == unique_labels(i));
    if length(idx_labels{i}) > maxnum_labels
        maxnum_labels = length(idx_labels{i});
    end
end

imsnew = imdstrain.Files;
labelnew = imdstrain.Labels;
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
        imsnew = vertcat(imsnew, imdstrain.Files{idx});
        labelnew = vertcat(labelnew, imdstrain.Labels(idx));
    end
end
imdstrain = imageDatastore(imsnew, 'FileExtension','.mat');
imdstrain.ReadFcn = @matimreader;
imdstrain.Labels = labelnew;

end

function draw_confusionmat(ytrue, ypred, classes, name)

for i=1:length(classes)
    if iscell(classes{i})
        temp = [];
        for l=1:length(classes{i})
            temp=strcat(temp,classes{i}{l});
        end
        classes{i}=temp;
    end
end

if sum(isundefined(ypred)) > 0
    return
end

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
name = strcat('Confusionmatrix_', name);
export_fig(gcf,name,'-painters','-png',150);
savefig(strcat(name,'.fig'));
close all
end

function r2 = draw_regression(ytrue, ypred, name)
mse = mean((ytrue - ypred).^2);
rmse = sqrt(mse);

figure1 = figure('Color',[1 1 1]);
axes1 = axes('Parent',figure1,...
    'Position',[0.13 0.11 0.403854166666667 0.815]);
hold(axes1,'on');
plot1 = plot(ytrue, ytrue,'-k','LineWidth',1);

% R^2
X = [ones(length(ytrue),1) ytrue];
b = X\ypred;
ycalc = X*b;
r2 = 1 - sum((ytrue - ycalc).^2)/sum((ytrue - mean(ytrue)).^2);

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

function stop = stopfun(info,N,mode)

stop = false;

    persistent bestValAccuracy
    persistent valLag
    persistent bestValRMSE
    
if strcmp(mode, 'classification')
    % Keep track of the best validation accuracy and the number of validations for which
    % there has not been an improvement of the accuracy.
    % Clear the variables when training starts.
    if info.State == "start"
        bestValAccuracy = 0;
        valLag = 0;

    elseif ~isempty(info.ValidationLoss)

        % Compare the current validation accuracy to the best accuracy so far,
        % and either set the best accuracy to the current accuracy, or increase
        % the number of validations for which there has not been an improvement.
        if info.ValidationAccuracy > bestValAccuracy
            valLag = 0;
            bestValAccuracy = info.ValidationAccuracy;
        else
            valLag = valLag + 1;
        end

        % If the validation lag is at least N, that is, the validation accuracy
        % has not improved for at least N validations, then return true and
        % stop training.
        if valLag >= N
            stop = true;
        end

    end
elseif strcmp(mode, 'regression')
    % Clear the variables when training starts.
    if info.State == "start"
        bestValRMSE = Inf;
        valLag = 0;

    elseif ~isempty(info.ValidationLoss)

        % Compare the current validation accuracy to the best accuracy so far,
        % and either set the best accuracy to the current accuracy, or increase
        % the number of validations for which there has not been an improvement.
        if info.ValidationRMSE < bestValRMSE
            valLag = 0;
            bestValRMSE = info.ValidationRMSE;
        else
            valLag = valLag + 1;
        end

        % If the validation lag is at least N, that is, the validation accuracy
        % has not improved for at least N validations, then return true and
        % stop training.
        if valLag >= N
            stop = true;
        end

    end
end

end