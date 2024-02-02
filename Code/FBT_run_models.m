%Giles Story London 2023

%Script to fit a set of models to probabilistic false belief task data, with
%options for ML, MAP or mixed effects optimisation

%Set up options as described in FBT_fit.m
options.fit='data';
options.doem=0;
options.doprior_init=1;
options.fitsjs='all';
options.session=1;
options.dostats=1;

%Run models with various combinations of parameters
for alpha=[1]
    for beta=1
        for delta=1
            for lambda=[1]
                model=[alpha beta delta lambda];
                [R] = FBT_fit(model, options);
            end
        end
    end
end

FBT_plotmodelfit

