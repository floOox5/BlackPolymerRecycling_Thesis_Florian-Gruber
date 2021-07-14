classdef hypercube
    % multi- / hyperspectral data set class
    properties
        filename        % name of the hypercube
        data            % 3d spektral data (samples x lines x bands)
        wl              % wavelength vector
        wl_unit         % unit of the wavelength vector
        length_unit     % unit of the hypercube size
        x_pxl           % pixel size of the sample side of the hypercube (perpendicular to scanning direction)
        y_pxl           % pixel size of the lines side of the hypercube (in scanning direction)
        pc              % PC from PCA
        meandata        % meanspek for PCA meancentering
        blob_im      % only used after blob tiling
        blob_data       % only used after blob tiling
        score_minmax    % for scaling the score values
        history         % processing history
        %pth             % path of the hypercube file
    end
    
    properties (Dependent)
        sb              % sum image of hypercube
        bands           % bands of the hypercube
        lines           % lines of the hypercube
        samples         % samples of the hypercubes
        scores          % score-values obtained by PCA
        mean_spek       % mean speak of the hypercube
        pth
    end
    
    methods
        % methods for dependent properties
        function sb = get.sb(self)                  
            % calculate sum image
            sb = mat2gray(sum(self.data,3));
        end
        function bands = get.bands(self)                  
            % calculate sum image
            bands = size(self.data,3);
        end
        function lines = get.lines(self)                  
            % calculate sum image
            lines = size(self.data,2);
        end
        function samples = get.samples(self)                  
            % calculate sum image
            samples = size(self.data,1);
        end
        function scores = get.scores(self)
            % calculates the scores of the hypercube
            spek = reshape(self.data,self.lines*self.samples,self.bands);
            if isempty(self.meandata) ~= 1
                spek = bsxfun(@minus, spek, self.meandata);
            end
            if ~(isempty(self.pc))
                scores = (spek * self.pc);
                if ~(isempty(self.score_minmax))
                    scores = (scores - repmat(self.score_minmax(1,1:size(scores,2)),[self.lines*self.samples 1]))./(repmat(self.score_minmax(2,1:size(scores,2))-self.score_minmax(1,1:size(scores,2)),[self.lines*self.samples, 1]));
                end
                idx = logical(mean(isnan(scores),2));
                scores(idx,:) = 0;
                scores = reshape(scores,self.samples, self.lines, size(self.pc,2));
            else
                scores = [];
            end
        end
        function mean_spek = get.mean_spek(self)
            mean_spek = reshape(self.data,self.samples*self.lines,self.bands);
            mean_spek = nanmean(mean_spek);
        end
        function pth = get.pth(self)
            [pth,~,~] = fileparts(which(strcat(self.filename,'.mat')));
        end
        % save function       
        function save_hc(self, name, pth)                      
            % save hypercube
            pth_org = cd;
            if nargin < 3 || isempty(pth)
                hc = self;
                if nargin > 1 && ~isempty(name)
                    self.filename = name;
                end
                save(self.filename,'hc', '-v7.3');
                cd(pth_org);
            else    
                if exist(pth,'dir') ~= 7
                    mkdir(pth)
                end
                
                addpath(genpath(pth));
                cd(pth)
                hc = self;
                if nargin > 1 && ~isempty(name)
                    self.filename = name;
                end
                save(self.filename,'hc');
                cd(pth_org);
            end
        end
        function save_envi(self,name)
        % writes the current hypercube to an .envi file with a .hdr header
        % file
        if nargin < 2
            name = strcat(self.filename,'_new');
        else
            name = strcat(self.filename, name);
        end
        envi_write(self.data,name,self.wl,self.wl_unit);
        end
        % plot and export functions
        function self = update_history(self, varargin)
            self.history{size(self.history,1)+1,1} = varargin{1};
            for i = 2:length(varargin)
                self.history{size(self.history,1),i} = varargin{i};
                self.history{size(self.history,1),i} = varargin{i};
            end
        end
        function pth_list = tile_hc(self, tile_mode, tile_size, pth, varargin)
            % tiles the hypercube in smaller hypercubes and saves them as
            % hypercube objects in the current folder or under 'pth' with
            % a unique position name (_tile-x-y). Hypercubes can be
            % restiched with the stitch function.
            if nargin < 4; pth = cd(); end; if isempty(pth) == 1; pth = cd(); end
            
            pth_list = {};
            
            if strcmp(tile_mode, 'parts') == 1
                x = floor(self.samples / tile_size(1));
                xnum = tile_size(1);
                y = floor(self.lines / tile_size(2));
                ynum = tile_size(2);
                for i = 1: xnum
                    for n = 1: ynum
                        data_ = self.data(1+(i-1)*x:i*x,1+(n-1)*y:n*y,:);
                        name = strcat(self.filename, '_tile-', num2str(i), '-', num2str(n));
                        hc = self;
                        hc.data = data_;
                        hc.filename = name;
                        %hc.pth = pth;
                        hc = hc.update_history('tiled', tile_mode, tile_size);
                        hc.save_hc(name, pth);
                        pth_list{size(pth_list,1)+1,1} = fullfile(pth, name);
                    end
                end             
            elseif strcmp(tile_mode, 'pixel') == 1
                xnum = floor(self.samples / tile_size(1));
                x = tile_size(1);
                ynum = floor(self.lines / tile_size(2));
                y = tile_size(2);
                for i = 1: xnum
                    for n = 1: ynum
                        data_ = self.data(1+(i-1)*x:i*x,1+(n-1)*y:n*y,:);
                        name = strcat(self.filename, '_tile-', num2str(i), '-', num2str(n));
                        hc = self;
                        hc.data = data_;
                        hc.filename = name;
                        hc = hc.update_history('tiled', tile_mode, tile_size);
                        hc.save_hc(name, pth);
                        pth_list{size(pth_list,1)+1,1} = fullfile(pth, name);
                    end
                end   
            else
                error('Incorrect tiling mode. Choose parts or pixel.');
            end
        end
        function pth_list = find_blobs(self, threshold, smallest_blobsize, remove_borderblobs, invert_image, data, range, pth)

            if nargin < 8; pth = cd(); end; if isempty(pth) == 1; pth = cd(); end
            if nargin < 7 || isempty(range); range = 1; end
            if nargin < 6 || isempty(data); data = 'scores'; end
            if nargin < 5 || isempty(invert_image); invert_image = 0; end
            if nargin < 4 || isempty(remove_borderblobs); remove_borderblobs = 1; end
            if nargin < 3 || isempty(smallest_blobsize); smallest_blobsize = 10; end
            
            pth_list = {};
            
            if strcmp(data, 'data')
                [~,idx1] = min(abs(self.wl - range(1)));
                [~,idx2] = min(abs(self.wl - range(end)));
                im = self.data(:,:,idx1:idx2);
            else
                im = self.(data)(:,:,range);
            end
            
            if size(im,3) > 1
                im = mean(im,3);
            end
            
            im = mat2gray(im);
                     
            if invert_image == 1
                im = imcomplement(im);
                threshold = 1-threshold;
            end
            if ischar(threshold) && strcmpi(threshold,'otsu') || isempty(threshold) || nargin < 2
                bw = imbinarize(im);
            elseif ischar(threshold) && strcmpi(threshold,'adaptive')
                bw = imbinarize(im,'adaptive');
            elseif isnumeric(threshold)
                bw = im2bw(im,threshold);
            else
                error('Incorrect threshold.');
            end

            bw = imfill(bw,'holes');
            bw=bwareaopen(bw,smallest_blobsize);
            if remove_borderblobs == 1
                bw = imclearborder(bw);
            end
            blob_image = bwlabel(bw);
            blobs = regionprops(blob_image,'BoundingBox','Centroid','Area','Perimeter',...
                'Solidity','Eccentricity');
            
            [rect, ~] = imOrientedBox(blob_image);
            for i=1:length(blobs)
                blobs(i).MaxFeret = rect(i, 3);
                blobs(i).MinFeret = rect(i, 4);
                blobs(i).Aspectratio = blobs(i).MinFeret/blobs(i).MaxFeret;
                blobs(i).Sphericity = 4*pi*blobs(i).Area/(blobs(i).Perimeter).^2;
                blobs(i).Roundness = (4*blobs(i).Area)/(pi*(blobs(i).MaxFeret).^2);
                blobs(i).Circularity = sqrt(4*blobs(i).Area/(pi*(blobs(i).MaxFeret).^2));
                blobs(i).Compactness = sqrt(((4*blobs(i).Area)/pi))/(blobs(i).MaxFeret);
            end
            prop = blobs;
            prop = rmfield(prop,'Centroid');
            prop = rmfield(prop,'BoundingBox');
            prop = rmfield(prop,'MinFeret');
            prop = rmfield(prop,'MaxFeret');            
            prop = rmfield(prop,'Area');
            prop = rmfield(prop,'Perimeter');
            prop = cell2mat(struct2cell(prop))';
            
            [l1,l2,l3] = size(self.data);

            for i = 1:length(blobs)
                mask = blob_image == i;
                data_ = reshape(self.data,l1*l2,l3);
                data_blob = double(reshape(bw,l1*l2,1));
                data_(~mask(:),:) = NaN;
                data_blob(~mask(:),:) = NaN;
                data_ = reshape(data_,l1,l2,l3);
                data_blob = reshape(data_blob,l1,l2,1);
                s1 = ceil(blobs(i).BoundingBox(1));
                s2 = s1+blobs(i).BoundingBox(3)-1;
                s3 = ceil(blobs(i).BoundingBox(2));
                s4 = s3+blobs(i).BoundingBox(4)-1;    
                data_ = data_(s3:s4,s1:s2,:);
                data_blob = data_blob(s3:s4,s1:s2,:);
                
                name = strcat(self.filename, '_blob-image-', num2str(i));

                hc = self;
                hc.data = data_;
                hc.blob_im = data_blob;
                hc.blob_data = prop(i,:);
                hc.filename = name;
                hc = hc.update_history('blob', threshold, smallest_blobsize, remove_borderblobs, invert_image, data, range);
                hc.save_hc(name, pth);
                pth_list{size(pth_list,1)+1,1} = fullfile(pth, name);
            end
        end
        function self = set_units(self, wl_unit, length_unit, varargin)
            % sets units for wavelength and size
            % defaults: 
            % wl_unit = nm
            % lateral_unit = mm
            
            if isempty(wl_unit)
                self.wl_unit = 'nm';
            else
                self.wl_unit = wl_unit;
            end
            if isempty(length_unit)
                self.length_unit = 'mm';
            else
                self.length_unit = length_unit;
            end
        end
        function self = set_pxlsize(self, x_pxl, y_pxl, varargin)
            % sets size of each pixel in x and y direction
            % units are defined by length_unit
            % defaults: 
            % x_pxl = y_pxl = 1
            
            if isempty(x_pxl)
                self.x_pxl = 1;
            else
                self.x_pxl = x_pxl;
            end
            if isempty(y_pxl)
                self.y_pxl = 1;
            else
                self.y_pxl = y_pxl;
            end
        end
        function self = cut_bands(self, band1, band2, varargin)
            % cut the hypercube wavelengths
            if isempty(band1) == 0
                [~, idx1] = min(abs(self.wl - band1));
            else
                idx1 = 1;
            end
            if isempty(band2) == 0
                [~, idx2] = min(abs(self.wl - band2));
            else
                idx2 = length(self.wl);
            end
            self.data = self.data(:,:,idx1:idx2);
            self.wl = self.wl(idx1:idx2);
            self = self.update_history('cut_bands', band1, band2);
        end
        function self = cut_hypercube(self, coordinates, show, varargin)
            % cuts the hypercube to x1:x2 and y1:y2
            % coordinates = [x1 x2 y1 y2]
            % if x or y are not given, the cutting are is selected
            % manually
            % show defines if the cutted hypercube is displayed (1 = yes /
            % 0 = no); default is 1
            if nargin < 3 || isempty(show); show = 1; end
            if iscell(coordinates) == 1; coordinates = coordinates{:}; end
            if iscell(show) == 1; show = show{:}; end
            
            if length(coordinates) ~= 4 
                [~, rect] = imcrop(self.sb);
                close all
                self.data = self.data(floor(rect(2)):floor(rect(2)+rect(4)),...
                    floor(rect(1)):floor(rect(1)+rect(3)),:);
                if show == 1; self.show_figure(self.wl); end
                self = self.update_history('cut_hypercube',floor(rect(1)),...
                    floor(rect(1)+rect(3)),floor(rect(2)),floor(rect(2)+rect(4)));
            else
                x1 = coordinates(1);x2 = coordinates(2);
                y1 = coordinates(3);y2 = coordinates(4);
                if x1 < 1; x1 = 1;end
                if x2 > self.lines || x2 < 1; x2 = self.lines;end
                if y1 < 1; y1 = 1;end
                if y2 > self.samples || y2 < 1; y2 = self.samples;end
                
                self.data = self.data(y1:y2,x1:x2,:);
                if show == 1; self.show_figure(self.wl); end
                self = self.update_history('cut_hypercube',x1,x2,y1,y2);
            end
        end
        function self = sgolay(self, polynom, points, deriv, varargin)
            % calculate savitky-golay-smoothing for each spectrum in the
            % hypercube
            % default values: polynom = 2; points = 5; deriv = 0
            
            % set default values
            if nargin < 2 || isempty(polynom); polynom = 2; end
            if nargin < 3 || isempty(points); points = 5; end
            if nargin < 4 || isempty(deriv); deriv = 0; end
            
            self.data = sgolay_fun(self.data, polynom, points, deriv);
            self.wl = self.wl(ceil(points/2):end-ceil(points/2));
            self = self.update_history('savitzky_golay', polynom, points, deriv);
        end
        function self = lnorm(self, norm, varargin)
            % divide each spectrum of the hypercube by his l-norm
            % default values: norm = 1
            
            % set default values
            if nargin < 2 || isempty(norm); norm = 1; end
            
            self.data = norm_fun(self.data, norm);
            self = self.update_history('lnorm', norm);
        end
        function self = snv(self, varargin)
            % calculate SNV-correction for each spectrum in the hypercube
            
            self.data = snv_fun(self.data);
            self = self.update_history('SNV');
        end
        function self = minmax(self, varargin)
            % scale each spectrum in the hypercube between min and max
            
            self.data = minmax_fun(self.data);
            self = self.update_history('MinMax');
            
        end
        function self = resample(self, x, y, z, varargin)
            % resamples the hypercube data (takes the mean of 'x' pixel in
            % x-direction, 'y' pixel ...
            temp = self.data;
            [s1, s2, s3] = size(temp);
            while mod(s1, x) ~= 0
                temp = temp(1:end-1,:,:);
                s1 = s1-1;
            end
            while mod(s2, y) ~= 0
                temp = temp(:,1:end-1,:);
                s2 = s2-1;
            end
            while mod(s3, z) ~= 0
                temp = temp(:,:,1:end-1);
                self.wl = self.wl(1:end-1);
                s3 = s3-1;
            end

            temp = reshape(temp, x, s2, []);
            temp = mean(temp,1);
            temp = reshape(temp, s1/x, s2, s3);
            s1 = s1/x;
            temp = reshape(temp, s1, y, []);
            temp = mean(temp,2);
            temp = reshape(temp, s1, s2/y, s3);
            s2 = s2/y;
            temp = permute(temp, [3 2 1]);
            temp = reshape(temp, z, s2, []);
            temp = mean(temp,1);
            temp = reshape(temp, s3/z, s2, s1);
            self.data = permute(temp,[3 2 1]);
            self.wl = mean(reshape(self.wl, z, s3/z));
            self = self.update_history('resample',x,y,z);
        end
        function self = meancenter(self, varargin)
            % meancenter each spectrum in the hypercube
            
            self.data = meancenter_fun(self.data);
            self = self.update_history('MeanCenter');
            
        end
        function self = medfilt(self,points, varargin)
            % medianfilter with 'points' number of points
            self.data = medfilt1(self.data, points, [], 3);
            self = self.update_history('MedianFilter', points);
        end
        function self = backcor(self,poly, th, fun, varargin)
            % medianfilter with 'points' number of points
            x=self.data;
            [s1, s2, s3] = size(x);
            x = reshape(x,s1*s2,s3);
            for i=1:s1*s2
                [bg(i,:),~,~] = backcor(self.wl,x(i,:),poly,th,fun);
            end
            x = x - bg;
            self.data = reshape(x, s1, s2, s3);
        end
        function self = remove_spikes(self, varargin)
            % removes spikes from spectra. Starts from the 1. band and
            % replaces the nth band with the(n-1)th band if n >
            % (n-1)*(1+threshold) or n < (n-1)*(1-threshold)
            temp = reshape(self.data, self.samples* self.lines, self.bands);
%             for i=1:size(temp,2)-1
%                 idx = temp(:,i+1) > temp(:,i)*(1+threshold) | temp(:,i+1) < temp(:,i)*(1-threshold);
%                 temp(idx,i+1) = temp(idx,i);
%             end
            
            for j=1: size(temp,2)-1
                idx = (temp(:,j+1) < temp(:,j) * (1/3)) | (temp(:,j+1) > temp(:,j) * 3);
                temp(idx, j+1) = temp(idx, j);
            end
            for j=1:size(temp,2)-2
                v(:, 1) = 0.5*(temp(:,j)+ temp(:,j+2));
                idx = ((temp(:, j+1) > 1.3 * v) & (temp(:, j + 2) > 0.7 * temp(:, j)) & (temp(:, j) > 0.7 * temp(:, j + 2))) | ((temp(:, j+1) < 0.7 * v) & (temp(:, j + 2) < 1.3 * temp(:, j)) & (temp(:, j) < 1.3 * temp(:, j + 2)));
                temp(idx, j+1) = v(idx, 1);
            end
            self.data = reshape(temp, self.samples, self.lines, self.bands);
            self = self.update_history('SpikeRemoval', []);
        end
        function self = apply_pca(self, numpc, varargin)
            hcens = varargin{end};
            self.pc = hcens.pc(:,1:numpc);
            self.meandata = hcens.mean_data;
            self.score_minmax = hcens.score_minmax;
            self = self.update_history('apply_pca', numpc);
        end
        function self = show_image(self, data, overlay, varargin)     
            % shows image of hypercube at the specified bands
            % If length(bands) > 3, show_figure calculates the sum image
            if nargin < 3
                overlay = [];
            end         
            h1 = figure;
            xData = [0 (self.lines-1)*self.x_pxl];
            yData = [0 (self.samples-1)*self.y_pxl];
                
            if size(data, 3) == 2
                data(:,:,3) = zeros(size(data(:,:,1)));
            elseif size(data, 3) > 3
                data = sum(data, 3);
            end

            if ~(isempty(overlay)) && iscategorical(overlay)
                data = labeloverlay(data, overlay, 'Transparency',0.75);
                imshow(data,'XData',xData,'YData',yData);
                set(gcf,'Position',get(0,'Screensize'));
                axis on
                xlabel(strcat('x [',self.length_unit,']'));
                ylabel(strcat('y [',self.length_unit,']'));
                set(gca,'FontSize',14,'FontWeight','bold');
                figure(h1)
            elseif ~(isempty(overlay))
                imshow(data,'XData',xData,'YData',yData);
                red = cat(3, ones(size(data,1),size(data,2)),zeros(size(data,1),size(data,2)),zeros(size(data,1),size(data,2)));
                hold on
                h = imshow(red,'XData',xData,'YData',yData);
                hold off
                set(h, 'AlphaData', overlay);
                set(gcf,'Position',get(0,'Screensize'));
                axis on
                xlabel(strcat('x [',self.length_unit,']'));
                ylabel(strcat('y [',self.length_unit,']'));
                set(gca,'FontSize',14,'FontWeight','bold');
                figure(h1)
            else
                imshow(data,'XData',xData,'YData',yData);
                set(gcf,'Position',get(0,'Screensize'));
                axis on
                xlabel(strcat('x [',self.length_unit,']'));
                ylabel(strcat('y [',self.length_unit,']'));
                set(gca,'FontSize',14,'FontWeight','bold');
                figure(h1)
            end 
        end
        function self = export_image(self, data, range, scale, overlay, dpi, pth, name, varargin)     
            % exports image of hypercube at the specified bands
            % If length(bands) > 3, show_figure calculates the sum image
            if nargin < 8 || isempty(name)
                name = strcat('_',data, '_', strrep(num2str(range),'  ','-'));
            end
            if nargin < 7 || (isempty(pth))
                org_pth = cd;
                
            else
                org_pth = cd;
                cd(pth);
            end
            if nargin < 6; dpi = 100; end; if isempty(dpi) == 1; dpi = 100; end
            if nargin < 5; overlay = []; end
            if nargin < 4; scale = 0; end
            if nargin < 3; error('Not enough input arguments'); end
            if isnumeric(data)
                if scale == 0
                    data_ = data;
                elseif scale == 1
                    for i = 1:size(data,3)
                        data_(:,:,i) = mat2gray(data(:,:,i));
                    end
                end
                self.show_image(data_, overlay);
            elseif strcmp(data, 'data')
                for n = 1:length(range)
                    [~,idx(n)] = min(abs(self.wl - range(n)));
                end
                data_ = self.data(:,:,idx);
                self.show_image(data_, overlay);
            else
                data_ = self.(data)(:,:,range);
                if scale == 0
                elseif scale == 1
                    for i = 1:size(data_,3)
                        data_(:,:,i) = mat2gray(data_(:,:,i));
                    end
                elseif scale ~=0 && scale ~= 1
%                     for i=1:size(data_,3)
%                         data_(:,:,i) = (data_(:,:,i)-scale)./(1-scale-scale);
%                     end
                    data_=imadjust(data_,[scale scale scale; 1-scale 1-scale 1-scale],[0 0 0; 1 1 1]);
                end
                self.show_image(data_, overlay);
            end
            resolution = strcat('-r',num2str(dpi));
            name = strcat(self.filename,name);
            export_fig(gcf,name,'-painters','-png',resolution);
            savefig(strcat(name,'.fig'));
            close all
            cd(org_pth);
        end
        function self = write_image(self, data, range, pth, varargin)     
            % write image of hypercube at the specified bands as plain .png file
            % If length(bands) > 3, show_figure calculates the sum image
            if nargin == 3 && ~(isempty(pth))
                org_pth = cd;
                cd(pth);
            end
            if length(range) == 1 || length(range) == 3
                for i = 1:length(range)
                    data_(:,:,i) = mat2gray(self.(data)(:,:,range(i)));
                end
            else
                data_ = mat2gray(sum(self.(data)(:,:,range),3));
            end
            imwrite(data_,strcat(self.name,'.png')); 
            cd(org_pth);
        end
        function show_speks(self,random, varargin)
            % shows the sum-image of the hypercubes
            % if 'random' ist set, 'random' spectra are choosen from the
            % hypercube and plotted
            % if random is not set, the user chooses spectra from the
            % hypercube
            im = self.show_sumimage(self.wl); 
            color = [];
            if nargin < 2
                [x, y] = getpts;
            else
                x = randi(size(im,2),random);
                y = randi(size(im,1),random);
            end
            x = x./self.x_pxl;
            y = y./self.y_pxl;
            
            for n=1:length(x)
                color(n,:) = rand(1,3);
                speks(:,n)=self.data(round(y(n)),round(x(n)),:);
                im = insertMarker(im,[round(x(n)) round(y(n))],'color',color(n,:),'Size',2);
            end
            close all
            
            figure(1)
            imshow(im)
            set(gcf,'Position',get(0,'Screensize'));
            axis on
            xlabel(strcat('x [',self.length_unit,']'));
            ylabel(strcat('y [',self.length_unit,']'));
            set(gca,'FontSize',14,'FontWeight','bold');
            
            figure2 = figure('Color',[1 1 1]);
            axes1 = axes('Parent',figure2,'Position',[0.13 0.11 0.3996875 0.815]);
            hold(axes1,'on');

            for i=1:size(speks,2)
                plot1 = plot(self.wl,speks(:,i),'LineWidth',1,'Color',color(i,:),'Parent',axes1);
                hold on
            end
            ylabel('Intensity');
            xlabel(self.length_unit);
            xlim(axes1,[min(self.wl) max(self.wl)]);
            box(axes1,'on');
            set(axes1,'FontSize',14,'FontWeight','bold');
            set(gcf,'Position',get(0,'Screensize'));
        end  
    end
    methods(Static)
        function hc = hypercube(data, wl, name, pth, pc, meandata, score_minmax, pxlunits, pxlsize)
        	if nargin == 0
                hc = hc;
                return
            end
            hc.history{1,1} = 'created:';
            hc.history{1,2} = clock;
            hc.data = data;                
            if nargin > 1
                hc.wl = wl;
            end
            if nargin > 2
                if isempty(name) == 1
                    hc.filename = 'hypercube';
                else
                    hc.filename = name;
                end         
            end
            if nargin > 3
                if ~(isempty(pth))
                    cd(pth)
                end         
            end
            addpath(genpath(cd));
            if nargin > 4
                hc.pc = pc;
            else
                hc.pc = [];
            end
            if nargin > 5
                hc.meandata = meandata;
            else
                hc.meandata = [];
            end
            if nargin > 6
                hc.score_minmax = score_minmax;
            else
                hc.score_minmax = [];
            end
            if nargin > 7
                hc = hc.set_units(pxlunits(1),pxlunits(2));
            else
                hc = hc.set_units([],[]);
            end           
            if nargin > 8
                if length(pxlsize) > 1
                    hc = hc.set_pxlsize(pxlsize(1), pxlsize(2));
                else
                    hc = hc.set_pxlsize(pxlsize, pxlsize);
                end
            else
                hc = hc.set_pxlsize([], []);
            end
        end
        function hc = hc_load(file_name)    % load hypercube from .envi file
            if nargin == 0
                [file_name, pathname]=uigetfile('*','Hypercube Wählen','Multiselect','Off');
                cd(pathname);
            end
            [data, wl, name, pth] = read_hc(file_name);
            data = permute(data, [2 1 3]);
            hc = hypercube(data, wl, name, pth);
        end
    end
end

%--------------------------------------------------------------------------
% supporting functions

function [data, wl, fn, pth] = read_hc(file_name)

[pth,fn,ext] = fileparts(file_name);
if strcmp(ext,'.envi') == 1
    hdr = strcat(file_name,'.hdr');
elseif strcmp(ext,'.hdr') == 1
    hdr = file_name;
    file_name = fn;
else
    error('Wrong file extension.');
end
if isempty(pth)
    pth = cd;
end

fid = fopen(hdr,'r');

while 1
     line = fgetl(fid);
     if ~ischar(line), break, end
%     switch lower(line)
         if not(isempty(strmatch('lines = ',line)))
             s = textscan(line,'lines = %s');
             lines = str2double(cell2mat(s{1}));
         end
         if not(isempty(strmatch('samples = ',line)))
             s = textscan(line,'samples = %s');
             samples = str2double(cell2mat(s{1}));
         end
         if not(isempty(strmatch('bands = ',line)))
             s = textscan(line,'bands = %s');
             bands = str2double(cell2mat(s{1}));
         end
         if not(isempty(strmatch('data type = ',line)))
             s = textscan(line,'data type = %s');
             dattype = str2double(cell2mat(s{1}));
         end
         if not(isempty(strmatch('wavelength = {',line)))
             s = textscan(line,'wavelength = {%s');
             s = textscan(cell2mat(s{1}),'%f','delimiter',',');
             wl=s{1};
         end
         if not(isempty(strmatch('data offset = ',line)))
             s = textscan(line,'data offset = %s');
             subfactor = str2double(cell2mat(s{1}));
         end
         if not(isempty(strmatch('scale factor = ',line)))
             s = textscan(line,'scale factor = %s');
             scalfactor = str2double(cell2mat(s{1}));
         end         
end
clear line;

fclose(fid);
 
% Daten auslesen
fid = fopen(file_name,'r');
switch dattype
     case 1
         file = fread(fid,'*uint8');
     case 2 
         file = fread(fid,'*int16'); 
     case 3
         file = fread(fid,'*int32');
     case 4 
         file = fread(fid,'*single');
     case 5
         file = fread(fid,'double=>single');
     case 12
         file = fread(fid,'*uint16');
     case 13
         file = fread(fid,'*uint32');
     case 14
         file = fread(fid,'int64=>int32');
     case 15
         file = fread(fid,'uint64=>uint32');
     otherwise
         file = fread(fid,'*char');
end

fclose(fid);   
fprintf('\nFile %s',file_name)
fprintf(' erfolgreich eingelesen \n');
 
data = single((reshape(file,samples,bands,lines)));
clear file

data = permute(data,[1,3,2]);
 
if dattype==12
    data=(data./scalfactor)+subfactor;
end

fn = strtok(fn,'.');

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
function [hc]=meancenter_fun(hc)
if ndims(hc)==2
    [~,s2]=size(hc);
    mean_hc=mean(hc,2);
    hc=hc./repmat(mean_hc,[1,s2]);
elseif ndims(hc)==3
    [s1,s2,s3]=size(hc);
    hc=reshape(hc,s1*s2,s3);
    mean_hc=mean(hc,2);
    hc=hc./repmat(mean_hc,[1,s3]);
    hc=reshape(hc,s1,s2,s3);
end
end
function envi_write(hc,fname,wl,wlunit)
% write hc-file to envi
% file can be opened with imantoPro
% Parameters initialization
fname = strcat(fname,'.envi');
hc = rot90(hc,3);
hc=flip(hc,2);
im_size=size(hc);
im_size(3)=size(hc,3);
hc=permute(hc,[1,3,2]);

d=[4 1 2 3 12 13];
% Check user input
if ~ischar(fname)
    error('fname should be a char string');
end

cl1=class(hc);
if cl1 == 'double'
    img=single(hc);
else
    img=hc;
end
cl=class(img);
switch cl
    case 'single'
        t = d(1);
    case 'int8'
        t = d(2);
    case 'int16'
        t = d(3);
    case 'int32'
        t = d(4);
    case 'uint16'
        t = d(5);
    case 'uint32'
        t = d(6);
    case 'double'
        t = 12;
    otherwise
        error('Data type not recognized');
end
wfid = fopen(fname,'w');
if wfid == -1
    i=-1;
end
disp([('Writing ENVI image ...')]);
fwrite(wfid,img,cl);
fclose(wfid);

% Write header file

fid = fopen(strcat(fname,'.hdr'),'w');
if fid == -1
    i=-1;
end

fprintf(fid,'%s \n','ENVI');
fprintf(fid,'%s %i \n','bands =',im_size(3));
fprintf(fid,'%s %i \n','lines =',im_size(2));
fprintf(fid,'%s %i \n','samples =',im_size(1));
fprintf(fid,'%s %i \n','data type =',t);
fprintf(fid,'%s %i \n','byte order =',0);
fprintf(fid,'%s %i \n','header offset =',0);
fprintf(fid,'%s %i \n','data offset =',0);
fprintf(fid,'%s %i \n','scale factor =',1);
fprintf(fid,'%s \n','interleave = BIL');
fprintf(fid,'%s %s \n','wavelength units =',wlunit);
fprintf(fid,'wavelength = {');
for i=1:length(wl)-1;fprintf(fid,'%0.1f,',wl(i));end
fprintf(fid,'%0.1f',wl(i+1));
fprintf(fid,'}');
fclose(fid);
end
