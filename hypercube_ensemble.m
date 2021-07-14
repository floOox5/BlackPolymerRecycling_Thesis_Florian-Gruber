classdef hypercube_ensemble
    % object that holds hypercube-objects
    % allows manipulation of all hypercube objects
    % allows further processing options
    
    properties
        hypercube_list              % pth to all stored hypercube objects
        filename                    % name of the hypercube ensemble
        pc                          % PCs of the hypercube ensemble
        mean_data                   % mean of all spectra in the hypercube; used for mean centering before PCA
        explained                   % explained variance by PCA
        gTruth                      % Label ground truth from ImageLabeler
        score_minmax                % minimum and maximum values of the scorevalues of the ensemble hypercubes
        history                     % processing history
        function_queue              % function queue
        pth
    end
    properties (Dependent)
    end
    
    methods
        function self = update_history(self, varargin)
            self.history{size(self.history,1)+1,1} = varargin{1};
            for i = 2:length(varargin)
                self.history{size(self.history,1),i} = varargin{i};
                self.history{size(self.history,1),i} = varargin{i};
            end
        end
        function save_ensemble(self, name, pth)                      
            % save hypercube
            pth_org = cd;
            hc_ensemble = self;
            if nargin > 1
                self.filename = name;
            end
            if nargin > 2
                cd(pth)
            end
            save(self.filename,'hc_ensemble');
            cd(pth_org);
        end
        function self = add_hc(self, hc_list)
            % adds 'hypercubes' to the 'hypercube_ensemble'
            if nargin < 2 || isempty(hc_list)
                [filename, pthname] = uigetfile('*','Dateien Wählen','Multiselect','On');
                celldata = cellstr(filename);
                [~,yc] = size(celldata);
                for i=1:yc
                    file_name = char(celldata(i));
                    cd(pthname);
                    [~,fn,ext] = fileparts(file_name);
                    if strcmp(ext,'.mat') == 1
                        load(file_name, 'hc');
                        for n = 1: size(self.history,1)-1
                            args = {};
                            for l = 1: size(self.history,2)-1
                                args{l} = self.history{n+1,l+1};
                            end
                            hc = hc.(self.history{n+1,1})(args{:},self);
                        end
                        hc.save_hc(strcat(fn,'_new.mat')); 
                        hc_list{i,1} = strcat(pthname,strcat(fn,'_new.mat'));
                    else
                        hc = hypercube.hc_load(file_name);
                        hc.save_hc;
                        fn = strtok(fn,'.');
                        for n = 1: size(self.history,1)-1
                            args = {};
                            for l = 1: size(self.history,2)-1
                                args{l} = self.history{n+1,l+1};
                            end
                            hc = hc.(self.history{n+1,1})(args{:},self);
                        end
                        hc.save_hc(strcat(fn,'_new.mat')); 
                        hc_list{i,1} = strcat(pthname,strcat(fn,'_new.mat'));
                    end  
                end
            else
                for i = 1: size(hc_list,1)
                    load(hc_list{i,1},'hc')
                    [pthname,fn,ext] = fileparts(hc_list{i,1});
                    if strcmp(ext,'.mat') == 1
                        load(hc_list{i,1}, 'hc');
                        for n = 1: size(self.history,1)-1
                            args = {};
                            for l = 1: size(self.history,2)-1
                                args{l} = self.history{n+1,l+1};
                            end
                            hc = hc.(self.history{n+1,1})(args{:},self);
                        end
                        hc.save_hc(strcat(fn,'_new.mat')); 
                        hc_list{i,1} = strcat(pthname,strcat(fn,'_new.mat'));
                    else
                        hc = hypercube.hc_load(hc_list{i,1});
                        hc.save_hc;
                        fn = strtok(fn,'.');
                        for n = 1: size(self.history,1)-1
                            args = {};
                            for l = 1: size(self.history,2)-1
                                args{l} = self.history{n+1,l+1};
                            end
                            hc = hc.(self.history{n+1,1})(args{:},self);
                        end
                        hc.save_hc(strcat(fn,'_new.mat')); 
                        hc_list{i,1} = strcat(pthname,strcat(fn,'_new.mat'));
                    end 
                end
            end
            t = ismember(hc_list,self.hypercube_list);
            if sum(t) ~= 0
                hc_list{t,:} = [];
            end
            hc_list=hc_list(~cellfun('isempty',hc_list));
            self.hypercube_list = vertcat(self.hypercube_list, hc_list);
        end
        function self = clear_hc(self, num)
            if num ~= 0
                for i = 1:length(num)
                    self.hypercube_list{num(i)} = [];
                    self.hypercube_list=self.hypercube_list(~cellfun('isempty',self.hypercube_list));
                end
            else
                self.hypercube_list = [];
            end
        end
        function history = show_history(self)
            history = self.history;
        end
        function queue = show_queue(self)
            queue = self.function_queue;
        end
        function self = clear_queue(self)
            self.function_queue = {};
            fprintf('Function queue cleared.\n')
        end
        % ensemble functions
        function self = ensemble_pca(self, numspeks, center)
            % loads each hypercube in the ensemble
            % selects 'numspeks' spectra at random from each hypercube and applies PCA.
            % If 'numspek' < 1 it is seen as portion of the spectra of each
            % hypercube. Default = 0.5.
            % If 'center' = 1 the spectra are meancentered before the PCA.
            % Default = 0;
            if nargin < 3; center = 0; end
            if nargin < 2; numspeks = 0.5; end
            data = [];
            for i = 1:numel(self.hypercube_list)
                hc = load(self.hypercube_list{i});
                
                fprintf('\nHypercube %i of %i read \n',i, numel(self.hypercube_list))
                
                tmp = fieldnames(hc);
                hc = hc.(tmp{1});
%                 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                 if length(hc.wl)>191
%                     hc.data = hc.data(:,:,2:end-3);
%                     hc.wl=hc.wl(2:end-3);
%                     hc.save_hc(hc.filename,hc.pth);
%                 end                    
                
                temp = reshape(hc.data,hc.lines*hc.samples,hc.bands);
                idx = logical(mean(isnan(temp),2)) | mean(temp,2) == 0;
                temp(idx,:)= [];
                idx = randperm(size(temp,1));
                if numspeks < 1
                    idx = idx(1:floor(length(idx)*numspeks));
                elseif numspeks > 1
                    idx = idx(1:numspeks);
                else
                end
                data = vertcat(data, temp(idx,:));
            end
            
            if center == 1
                self.mean_data = mean(data,1);
                data = bsxfun(@minus,data,self.mean_data);
                fprintf('\nStarting PCA.\n'); tic;
                [self.pc,scores,~,~,self.explained,~] = pca(data, 'Centered','off');
                %self.scores = scores;
                self.score_minmax(1,:) = min(scores);
                self.score_minmax(2,:) = max(scores);
                fprintf('\nPCA finished. Duration: %f\n',toc)
            else
                fprintf('\nStarting PCA.\n'); tic;
                [self.pc,scores,~,~,self.explained,~] = pca(data, 'Centered','off');
                %self.scores = scores;
                self.score_minmax(1,:) = min(scores);
                self.score_minmax(2,:) = max(scores);
                fprintf('\nPCA finished. Duration: %f\n',toc)
                self.mean_data = [];
            end
            %self = self.update_history('PCA', numspeks, center);
        end
        function self = apply_pca(self, numpc)
            for i = 1:numel(self.hypercube_list)
                hc = load(self.hypercube_list{i});
                                
                fprintf('\nHypercube %i of %i read \n',i, numel(self.hypercube_list))
                
                tmp = fieldnames(hc);
                hc = hc.(tmp{1});
                hc.pc = self.pc(:,1:numpc);
                hc.meandata = self.mean_data;
                hc.score_minmax = self.score_minmax;
                [pth_, name, ~] = fileparts(self.hypercube_list{i});
                %hc.save_hc(name, pth_);
                hc.save_hc(name, 'E:\Projekte\Auswertung_flexRec\Daten');
                self.hypercube_list{i} = fullfile(pth_, name);

            end
            self = self.update_history('PCA', numpc);
        end
        function self = tile_hcs(self, tile_mode, tile_size, pth)
            if nargin < 4; pth = cd(); end; if isempty(pth) == 1; pth = cd(); end
            
            pth_list = {};
            
            for i = 1:numel(self.hypercube_list)
                hc = load(self.hypercube_list{i});
                                
                fprintf('Hypercube %i of %i read \n',i, numel(self.hypercube_list))
                
                tmp = fieldnames(hc);
                hc = hc.(tmp{1});
                pth_list = vertcat(pth_list,hc.tile_hc(tile_mode, tile_size, pth));
            end
            self.hypercube_list = pth_list;
            self = self.update_history('tiled', tile_mode, tile_size);
        end
        function self = find_blobs(self, threshold, smallest_blobsize, remove_borderblobs, invert_image, data, range, pth)
            if nargin < 8; pth = cd(); end; if isempty(pth) == 1; pth = cd(); end
            
            pth_list = {};
            
            for i = 1:numel(self.hypercube_list)
                hc = load(self.hypercube_list{i});
                                
                fprintf('Hypercube %i of %i read \n',i, numel(self.hypercube_list))
                
                tmp = fieldnames(hc);
                hc = hc.(tmp{1});
                pth_list = vertcat(pth_list,hc.find_blobs( threshold, smallest_blobsize, remove_borderblobs, invert_image, data, range, pth));
            end
            self.hypercube_list = pth_list;
            self = self.update_history('find_blobs',  threshold, smallest_blobsize, remove_borderblobs, invert_image, data, range);
        end
        % functions for all hypercubes in the ensemble (same as for
        % 'hypercube' class; it is recommended to use 'apply_function'
        function self = queue(self, fun, varargin)
            % writes functions in the function queue
            % queue is excecuted with the execute function
            % possible functions to queue:
            % set_units: wl_unit, length_unit
            % set_pxlsize: x_pxl, y_pxl
            % cut_bands: band1, band2
            % cut_hypercube: coordinates, show
            % sgolay: polynom, points, deriv
            % lnorm: norm
            % snv: []
            % minmax: []
            % meancenter: []
            % medfilt: points
            % remove_spikes: threshold
            n = size(self.function_queue,1);
            self.function_queue{n+1,1} = fun;
            for i = 1: numel(varargin)
                self.function_queue{n+1,i+1}=varargin{i};
            end
            fprintf('\nFunction %s added to queue. \n',fun)
        end            
        function self = apply(self, overwrite, pth)
            % applies all function in the function_queue to all hypercubes
            % in the ensemble
            % 'path' is the location to save the changed hypercubes
            % If 'overwrite' = 1 existing hypercubes in 'path' with the
            % same name are overwritten. If 'overwrite' = 0 they are saved
            % with the addition '_new'. Default is 0;
            if nargin < 2; overwrite = 0; end; if isempty(overwrite) == 1; overwrite = 0; end
            
            for i = 1:numel(self.hypercube_list)
                hc = load(self.hypercube_list{i});
                                
                fprintf('Hypercube %i of %i read \n',i, numel(self.hypercube_list))
                
                tmp = fieldnames(hc);
                hc = hc.(tmp{1});
                for j=1:size(self.function_queue,1)
                    args = {};
                    for n = 1: size(self.function_queue,2)-1
                        args{n} = self.function_queue{j,n+1};
                    end
                    hc = hc.(self.function_queue{j,1})(args{:},self);
                    fprintf('Funtion %s applied. \n',self.function_queue{j,1})
                end
                [~, name, ~] = fileparts(self.hypercube_list{i});
                if nargin < 3 || isempty(pth); pth = hc.pth; end
                if overwrite == 0
                    hc.filename = strcat(name,'_new');
                    hc.save_hc(strcat(name,'_new'), pth);
                    self.hypercube_list{i} = fullfile(pth, strcat(name,'_new'));
                else
                    hc.save_hc(name, pth);
                    self.hypercube_list{i} = fullfile(pth, name);
                end
            end
            for i = 1: size(self.function_queue,1)
                self = self.update_history(self.function_queue{i,1},self.function_queue{i,2:end});
            end
            self = self.clear_queue;
        end
        % labeling
        function self = label_hcs(self, label)
            if nargin < 2 || (~strcmp(label,'folder') && ~strcmp(label, 'is') && ~strcmp(label, 'none'))
                label = [];
            end
                    
            for i = 1:numel(self.hypercube_list)
                hc = load(self.hypercube_list{i});
                                
                fprintf('Hypercube %i of %i read \n',i, numel(self.hypercube_list))
                
                tmp = fieldnames(hc);
                hc = hc.(tmp{1});
                lhc = labeled_hypercube(hc, label);              
                
                if strcmp(label, 'none')
                    lhc.save_hc(lhc.filename, lhc.pth)
                    continue
                end
                
                if i == 1
                    label_list{i} = num2str(lhc.label);
                end
                if ~strcmp(label_list{end},num2str(lhc.label))
                    label_list{length(label_list)+1} = num2str(lhc.label);
                end
                if ~isempty(label)
                    lhc.save_hc(lhc.filename, lhc.pth)
                end
            end
            if (strcmp(label,'none'))
                
            elseif ~(strcmp(label,'folder'))
                if isempty(label)
                    cd('ImageLabeling')
                    if exist(fullfile(pwd,'\PixelLabelData','dir')) == 7
                        rmdir('PixelLabelData','s')
                    end
                    uiwait(msgbox('Label your images and the press: "Export Labels" -> "Export to File" and close the ImageLabeler. The press the "Ok" button in the next window.'));
                    imageLabeler(pwd)
                    uiwait(msgbox('Labeling completed'));
                elseif strcmp(label, 'is')
                    if exist(fullfile(pwd,'\ImageLabeling','dir')) == 7
                        cd('ImageLabeling')
                    end
                end
               load('gTruth.mat');
                self.gTruth = gTruth;
                for i = 1:numel(self.hypercube_list)
                    hc = load(self.hypercube_list{i});

                    fprintf('Hypercube %i of %i read \n',i, numel(self.hypercube_list))

                    tmp = fieldnames(hc);
                    hc = hc.(tmp{1});
                    label = categorical(ones(hc.samples,hc.lines)*NaN);

                    for n = 1: size(self.gTruth.DataSource.Source,1)
                        filestr = self.gTruth.DataSource.Source{n};
                        [~, filestr, ~] = fileparts(filestr);
                        list(n) = strncmp(filestr,hc.filename,length(filestr)-8);%
                    end
                    o = find(list==1);

                    for n = 1:size(self.gTruth.LabelDefinitions,1)
                        if strcmp(self.gTruth.LabelDefinitions.Type(n),'Rectangle')
                            for l = 1:size(self.gTruth.LabelData.(self.gTruth.LabelDefinitions.Name{n}){o},1)
                                y = self.gTruth.LabelData.(self.gTruth.LabelDefinitions.Name{n}){o}(l,1);
                                x = self.gTruth.LabelData.(self.gTruth.LabelDefinitions.Name{n}){o}(l,2);
                                yl = self.gTruth.LabelData.(self.gTruth.LabelDefinitions.Name{n}){o}(l,3);
                                xl = self.gTruth.LabelData.(self.gTruth.LabelDefinitions.Name{n}){o}(l,4);
                                label(x:x+xl,y:y+yl) = self.gTruth.LabelDefinitions.Name{n};
                            end
                         elseif strcmp(self.gTruth.LabelDefinitions.Type(n),'PixelLabel')
                            [~, ~, ext] = fileparts(self.gTruth.LabelData.PixelLabelData{o});
                            if ~(isempty(ext))
                                im = imread(self.gTruth.LabelData.PixelLabelData{o});
                                label(im == self.gTruth.LabelDefinitions.PixelLabelID{n}) = categorical(cellstr(self.gTruth.LabelDefinitions.Name{n}));
                            end
                         end
                    end
                    lhc =  labeled_hypercube(hc, label);
                    lhc.save_hc(lhc.filename, lhc.pth)
                end
            else
                self.gTruth = label_list;
            end
            cd(self.pth);
        end
    end
    methods (Static)
        function hc_ensemble = hypercube_ensemble(hc_list, name)
            % creates hypercube ensemble from list of hypercube filepths
            if nargin == 0
                hc_ensemble = hc_ensemble;
                return
            end
            hc_ensemble.hypercube_list = hc_list;    % list of hypercubes
            hc_ensemble.history{1,1} = 'created:';   % processing history
            hc_ensemble.history{1,2} = clock;
            if nargin < 2
                hc_ensemble.filename = 'hypercube_ensemble';  % name of the hypercube file
            else
                hc_ensemble.filename = name;
            end
            hc_ensemble.pc = [];
            hc_ensemble.mean_data = [];
            hc_ensemble.explained = [];
            hc_ensemble.function_queue = [];
            hc_ensemble.gTruth = [];
            hc_ensemble.score_minmax = [];
            hc_ensemble.pth = cd;
            addpath(genpath(cd));
        end
        function hc_ensemble = ensemble_load
            %%%
            meanspeks = [];
            labels = {};
            %%%
            [filename, pthname] = uigetfile('*','Dateien Wählen','Multiselect','On');
            celldata = cellstr(filename);
            [~,yc] = size(celldata);
            for i=1:yc
                file_name = char(celldata(i));
                cd(pthname);
                [~,fn,ext] = fileparts(file_name);
                if strcmp(ext,'.mat') == 1
                    hc_list{i,1} = strcat(pthname,'\',file_name);
                else
                    hc = hypercube.hc_load(file_name);
                    %%%
%                     hc.data=hc.data(:,151:end,:);
%                     s = reshape(hc.data,hc.samples*hc.lines,hc.bands);
%                     s=mean(s);
%                     meanspeks = vertcat(meanspeks,s);
%                     labels=vertcat(labels,fn(13:19));
                    %%%
                    hc.save_hc;
                    fn = strtok(fn,'.');
                    hc_list{i,1} = strcat(pthname,'\',strcat(fn,'.mat'));
                end
            end
            %%%
%             save('MW-Spektren2.mat','meanspeks','labels');
            %%%
            hc_ensemble = hypercube_ensemble(hc_list);
        end
        function hc_ensemble = read_folder(pth, subfolder, file_ext)
            % read all files with file-extension file_ext in folder, 
            % checks if they contain hypercube objects and adds them to a 
            % hypercube ensemble (or reads them if .envi or .hdr
            % subfolder: also checks the subolders (1 = yes / 0 = no),
            % default = 0
            % file_ext = [.envi, .hdr, .mat], default = .mat
            if nargin < 2
                subfolder = 0;
            end
            if nargin < 3
                file_ext = '.mat';
            end
            
            cd(pth)
            orgpth = pth;
            foldername = fullfile(orgpth);
            pth = genpath(foldername);
            pth = strsplit(pth, ';');
            if subfolder == 0
                pth = pth(1,1);
            end
            
            hc_list={};
            
            for i = 1:size(pth,2)
                list{i,1} = dir(strcat(pth{1,i}, '\*', file_ext));
            end
            if strcmp(file_ext,'.mat') == 1
                l = 1;
                for i = 1:numel(list)
                    if isempty(list{i}) == 0
                        temp = list{i};
                        for j = 1: numel(temp)
                            hc_list{l,1} = fullfile(temp(j).folder,temp(j).name);
                            l = l+1;
                        end                       
                    end
                end
            else
                l = 1;
                for i = 1:numel(list)
                    if isempty(list{i}) == 0
                        temp = list{i};
                        for j = 1: numel(temp)
                            fn = fullfile(temp(j).folder,temp(j).name);
                            hc = hypercube.hc_load(fn);
                            hc.save_hc;
                            fn = strtok(temp(j).name,'.');
                            fn = strcat(fn,'.mat');
                            hc_list{l,1} = fullfile(temp(j).folder,fn);
                            l = l+1;
                        end
                    end
                end
            end
            if ~isempty(hc_list)
                hc_ensemble = hypercube_ensemble(hc_list);     
            else
                hc_ensemble = [];
            end
        end 
    end
    
end

