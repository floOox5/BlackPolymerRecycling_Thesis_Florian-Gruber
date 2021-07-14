classdef augment_opt

properties
    bg_excec = true;
    xref = false;
    yref = false;
    xtrans = [0 0];
    ytrans = [0 0];
    xscale = [1 1];
    yscale = [1 1];
    xshear = [0 0];
    yshear = [0 0];
    edtrans = [0 0];
    gauss = [0 0];
    rotation = [0 0];
end

methods
    function augopt = augment_opt(options)
        optnames = fieldnames(options);
        for i = 1:length(optnames)
            switch lower(num2str(optnames{i}))
                case 'backgroundexcecution'
                    augopt.bg_excec = options.(optnames{i});
                case 'xreflection'
                    augopt.xref = options.(optnames{i});
                case 'yreflection'
                    augopt.yref = options.(optnames{i});
                case 'xtranslation'
                    augopt.xtrans = options.(optnames{i});
                case 'ytranslation'
                    augopt.ytrans = options.(optnames{i});
                case 'xscale'
                    augopt.xscale = options.(optnames{i});
                case 'yscale'
                    augopt.yscale = options.(optnames{i});
                case 'xshear'
                    augopt.xshear = options.(optnames{i});
                case 'yshear'
                    augopt.yshear =  options.(optnames{i});
                case 'edtransformation'
                    augopt.edtrans = options.(optnames{i});
                case 'gaussnoise'
                    augopt.gauss = options.(optnames{i});
                case 'rotation'
                    augopt.rotation = options.(optnames{i});
                otherwise

            end
        end        
    end
end
end