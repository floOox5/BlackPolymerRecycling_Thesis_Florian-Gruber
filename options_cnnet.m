function options = options_cnnet()

options.Normalization = 'mean3';
options.ImageData = 'scores';
options.DataRange = 2:4;
options.ImageSize = [64 64];
options.ResizeMethod = 'resize';

options.BackgroundExcecution = true;
options.xreflection = false;
options.yreflection = false;
options.xtranslation = [0 0];
options.ytranslation = [0 0];
options.xscale = [1 1];
options.yscale = [1 1];
options.xshear = [0 0];
options.yshear = [0 0];
options.edtransformation = [0 0; 0 0]; %[3 7; 10 100];
options.gaussnoise = [0 0];
options.rotation = [0 360];

options.mode = 'classification';
options.net = [];
options.safeNet = 0;
options.layers = 4;
options.depth = 1;
options.filtermode = 'double';
options.filter = 32;
options.batchnormalization = 1;
options.relu = 'relu';
options.fullyconnect = 1;
options.sizefullyconnect = 50;
options.convolutionsize = [3 3];
options.convolutionstride = [1 1];
options.poolingmode = 'max';
options.poolingsize = [2 2];
options.poolingstride = [1 1];
options.dropout = [0 0 0 0 0.25];
    
options.momentum = 0.95;
options.learnrate = 0.01;
options.learnrateschedule = 'none';
options.learnratedropperiod = 100;
options.learnratedropfactor = 0.1;
options.l2regularization = 0.0005;
options.shuffle = 'every-epoch';
options.batchsize = 32;
options.plots = 'training-progress';
options.verbose = 0;
options.validationfrequency = 5;
options.validationpatience = 3;
options.maxepochs = 200;
options.datasplit = {0.5, 0.5, 0};
options.repeats = 10;
end